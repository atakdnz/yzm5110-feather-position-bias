"""FastV reproduction + spatial-coverage-constrained token selection on LLaVA-1.5-7B.

One prefill pass per (sample, method, budget). Layers [0, K) run on the full sequence;
at layer K the 576 image tokens are reduced to a budget k chosen by a selection rule;
layers [K, 32) run on the shorter sequence, keeping the ORIGINAL position ids of the
kept tokens (as in the official FastV code). The answer is read from the logits of the
last prompt position (single-token answers: A-OKVQA letter, MME yes/no).

Selection rules (all keep exactly k image tokens, so FLOPs after layer K are identical):
  fastv      top-k of head-averaged attention from the LAST token at layer K-1 (official code)
  fastv_avgq top-k of head-averaged attention RECEIVED, averaged over all query rows
             at layer K-1 (the criterion described in the paper text)
  cc4        coverage-constrained FastV (proposed): 24x24 grid split into 4x4 blocks; each
             block keeps its floor(k/16) highest-attention tokens (same score as fastv);
             leftover budget filled by global top attention
  cc2        same with 2x2 blocks (weaker constraint; ablation)
  grid       spatially uniform tokens, no attention (coverage-only control)
  rand{s}    uniform random tokens, seed s (attention-free control)
  fastvnr    top-k of last-token attention computed WITHOUT RoPE at layer K-1 (FEATHER's
             phi_-R de-biasing; added as an extra control after the main runs)
All hyper-parameters (K=2, budgets, block sizes) were fixed before any result was seen.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from PIL import Image

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REV = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
SYS = ("A chat between a curious human and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the human's questions.")
N_IMG, G = 576, 24
# Official FastV A-OKVQA template (src/FastV/inference/eval/inference_aokvqa.py)
AOK_TEMPLATE = ("\nAnalyse the image and choose the best answer for the following question:\n"
                "{question}\nOptions: {options}\nJust output the letter of the correct answer.")


def expand2square(img, bg):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    res = Image.new(img.mode, (s, s), bg)
    res.paste(img, ((s - w) // 2, (s - h) // 2))
    return res


class Runner:
    def __init__(self, K=2, dtype=torch.float16):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REV)
        self.tok = self.proc.tokenizer
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=REV, torch_dtype=dtype, attn_implementation="sdpa").cuda().eval()
        m = self.model
        if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
            self.lm, self.head = m.language_model.model, m.language_model.lm_head
        else:  # newer transformers layout
            self.lm, self.head = m.model.language_model, m.lm_head
        tc = m.config.text_config
        self.nh, self.nkv = tc.num_attention_heads, tc.num_key_value_heads
        self.hd = tc.hidden_size // tc.num_attention_heads
        self.K = K
        self.bg = tuple(int(x * 255) for x in self.proc.image_processor.image_mean)
        self.img_tok = m.config.image_token_index
        self.vlayer = m.config.vision_feature_layer
        # block index tables for coverage-constrained selection
        r = torch.arange(N_IMG) // G
        c = torch.arange(N_IMG) % G
        self.blocks = {}
        for B in (2, 4):
            bs = G // B
            bid = (r // bs) * B + (c // bs)
            self.blocks[B] = torch.stack([(bid == b).nonzero().squeeze(1) for b in range(B * B)]).cuda()

    # ---------- inputs ----------
    def tokenize(self, user_text):
        """Mimics LLaVA's tokenizer_image_token: [BOS] chunk0 <image>*576 chunk1."""
        prompt = f"{SYS} USER: <image>\n{user_text} ASSISTANT:"
        c0, c1 = prompt.split("<image>")
        ids0 = self.tok(c0).input_ids  # includes BOS
        ids1 = self.tok(c1).input_ids[1:]  # drop BOS
        ids = ids0 + [self.img_tok] * N_IMG + ids1
        return torch.tensor([ids], device="cuda"), len(ids0)

    @torch.no_grad()
    def embed(self, image, user_text):
        ids, img_start = self.tokenize(user_text)
        img = expand2square(image.convert("RGB"), self.bg)
        pv = self.proc.image_processor(images=img, return_tensors="pt").pixel_values.cuda().half()
        vo = self.model.vision_tower(pv, output_hidden_states=True)
        f = vo.hidden_states[self.vlayer][:, 1:]
        f = self.model.multi_modal_projector(f)
        emb = self.model.get_input_embeddings()(ids).clone()
        emb[0, img_start:img_start + N_IMG] = f[0].to(emb.dtype)
        return emb, img_start, ids

    # ---------- transformer pieces ----------
    def run_layers(self, h, pos, a, b):
        pe = self.lm.rotary_emb(h, pos)
        for layer in self.lm.layers[a:b]:
            out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe, use_cache=False)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def logits_last(self, h):
        return self.head(self.lm.norm(h[:, -1:]))[:, 0].float()

    def layer_attention(self, layer, h, pos):
        """Head-averaged softmax attention of `layer` on input h (B=1). Returns
        (last-token row, mean over all query rows, last-token row computed WITHOUT RoPE)
        as float32 vectors of length L."""
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        x = layer.input_layernorm(h)
        B, L, _ = x.shape
        q = layer.self_attn.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = layer.self_attn.k_proj(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        s_nr = (q[:, :, -1:].float() @ k.float().transpose(-1, -2)) / math.sqrt(self.hd)
        p_nr = s_nr.softmax(-1).mean(1)[0, 0]  # RoPE-free score (as in FEATHER's phi_-R)
        cos, sin = self.lm.rotary_emb(x, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.hd)
        s.masked_fill_(torch.triu(torch.ones(L, L, dtype=torch.bool, device=s.device), 1), float("-inf"))
        p = s.softmax(-1).mean(1)[0]  # (L, L), mean over heads
        return p[-1], p.mean(0), p_nr

    # ---------- selection rules ----------
    def select(self, method, k, a_last, a_avg, sample_idx, a_nr=None):
        if method == "fastv":
            return a_last.topk(k).indices
        if method == "fastv_avgq":
            return a_avg.topk(k).indices
        if method == "fastvnr":  # control added after the main runs: RoPE-free last-token score
            return a_nr.topk(k).indices
        if method.startswith("cc"):
            B = int(method[2:])
            blk = self.blocks[B]
            q = k // (B * B)
            chosen = torch.zeros(N_IMG, dtype=torch.bool, device=a_last.device)
            if q > 0:
                top = blk.gather(1, a_last[blk].topk(q, dim=1).indices)
                chosen[top.flatten()] = True
            rem = k - int(chosen.sum())
            if rem > 0:
                a2 = a_last.clone()
                a2[chosen] = float("-inf")
                chosen[a2.topk(rem).indices] = True
            return chosen.nonzero().squeeze(1)
        if method == "grid":
            return torch.linspace(0, N_IMG - 1, k, device=a_last.device).round().long()
        if method.startswith("rand"):
            s = int(method[4:])
            g = torch.Generator().manual_seed(1000003 * s + sample_idx)
            return torch.randperm(N_IMG, generator=g)[:k].to(a_last.device)
        raise ValueError(method)

    # ---------- full pipeline for one sample ----------
    @torch.no_grad()
    def prefill_all(self, image, user_text, methods, budgets, sample_idx):
        """Returns dict name -> logits(vocab) and dict (method,k) -> selected indices.
        Layers [0,K) are shared (identical for all methods)."""
        emb, img_start, ids = self.embed(image, user_text)
        L = emb.shape[1]
        pos = torch.arange(L, device="cuda")[None]
        K = self.K
        h = self.run_layers(emb, pos, 0, K - 1)
        a_last, a_avg, a_nr = self.layer_attention(self.lm.layers[K - 1], h, pos)
        a_last = a_last[img_start:img_start + N_IMG]
        a_avg = a_avg[img_start:img_start + N_IMG]
        a_nr = a_nr[img_start:img_start + N_IMG]
        h = self.run_layers(h, pos, K - 1, K)  # output of layer K-1 = input to layer K
        out = {"full": self.logits_last(self.run_layers(h, pos, K, len(self.lm.layers)))[0]}
        sels = {}
        pre = torch.arange(img_start, device="cuda")
        post = torch.arange(img_start + N_IMG, L, device="cuda")
        for k in budgets:
            keeps = []
            for mth in methods:
                sel = self.select(mth, k, a_last, a_avg, sample_idx, a_nr).sort().values
                assert sel.numel() == k and sel.unique().numel() == k, (mth, k)
                sels[(mth, k)] = sel
                keeps.append(torch.cat([pre, img_start + sel, post]))
            keep = torch.stack(keeps)  # (M, L')
            hb = h[0][keep]  # (M, L', D)
            hb = self.run_layers(hb, keep, K, len(self.lm.layers))
            lg = self.logits_last(hb)
            for i, mth in enumerate(methods):
                out[f"{mth}@{k}"] = lg[i]
        return out, sels, a_last.cpu().numpy(), emb, ids

    @torch.no_grad()
    def timed_prefill(self, image, user_text, method, k, sample_idx):
        """End-to-end prefill (vision encoder + projector + selection + LLM), batch 1."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb, img_start, _ = self.embed(image, user_text)
        L = emb.shape[1]
        pos = torch.arange(L, device="cuda")[None]
        K = self.K
        nL = len(self.lm.layers)
        if method == "full":
            h = self.run_layers(emb, pos, 0, nL)
        else:
            h = self.run_layers(emb, pos, 0, K - 1)
            needs_attn = method.startswith("fastv") or method.startswith("cc")
            if needs_attn:
                a_last, a_avg, _ = self.layer_attention(self.lm.layers[K - 1], h, pos)
                a_last, a_avg = a_last[img_start:img_start + N_IMG], a_avg[img_start:img_start + N_IMG]
            else:
                a_last = a_avg = torch.zeros(N_IMG, device="cuda")
            h = self.run_layers(h, pos, K - 1, K)
            sel = self.select(method, k, a_last, a_avg, sample_idx).sort().values
            keep = torch.cat([torch.arange(img_start, device="cuda"), img_start + sel,
                              torch.arange(img_start + N_IMG, L, device="cuda")])
            h = self.run_layers(h[:, keep], keep[None], K, nL)
        lg = self.logits_last(h)
        tok = lg.argmax(-1).item()
        torch.cuda.synchronize()
        return time.perf_counter() - t0, tok


def letter_ids(tok, letters):
    out = {}
    for Lt in letters:
        cands = {tok.convert_tokens_to_ids("▁" + Lt), tok.convert_tokens_to_ids(Lt)}
        out[Lt] = [c for c in cands if c is not None and c != tok.unk_token_id]
    return out


def img_of(ex):
    return ex["ds"][ex["idx"]]["image"]


def load_task(task, limit):
    from datasets import load_dataset
    if task == "aokvqa":
        ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
        items = []
        for i, ex in enumerate(ds.remove_columns(["image"])):
            opts = " ".join(f"({chr(65 + j)}) {c}" for j, c in enumerate(ex["choices"]))
            items.append(dict(idx=i, qid=ex["question_id"], ds=ds,
                              text=AOK_TEMPLATE.format(question=ex["question"], options=opts),
                              answer=chr(65 + ex["correct_choice_idx"]), n_opt=len(ex["choices"])))
    elif task == "mme":
        ds = load_dataset("lmms-lab/MME", split="test")
        items = []
        for i, ex in enumerate(ds.remove_columns(["image"])):
            items.append(dict(idx=i, qid=ex["question_id"], ds=ds, text=ex["question"],
                              answer=ex["answer"].strip().lower(), category=ex["category"]))
    else:
        raise ValueError(task)
    if limit:
        items = items[:limit]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="aokvqa", choices=["aokvqa", "mme"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--budgets", default="288,144,58,29")  # R = 50%, 75%, 90%, 95% of 576
    ap.add_argument("--methods", default="fastv,fastv_avgq,cc4,cc2,grid,rand0,rand1,rand2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--latency", type=int, default=0, help="if >0: timing run on this many samples")
    ap.add_argument("--sanity", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",")]
    methods = args.methods.split(",")
    R = Runner(K=args.K)
    tok = R.tok
    items = load_task(args.task, args.limit)
    meta = dict(model=MODEL_ID, task=args.task, n=len(items), K=args.K, budgets=budgets, methods=methods,
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    import transformers, datasets
    meta.update(transformers=transformers.__version__, datasets=datasets.__version__)
    try:
        from huggingface_hub import snapshot_download
        meta["model_revision"] = os.path.basename(snapshot_download(MODEL_ID, revision=REV, local_files_only=True))
    except Exception as e:  # noqa
        meta["model_revision"] = f"unknown ({e})"
    print(json.dumps(meta), flush=True)

    if args.task == "aokvqa":
        lid = letter_ids(tok, "ABCD")
    else:
        lid = {"yes": letter_ids(tok, ["Yes", "yes"])["Yes"] + letter_ids(tok, ["yes"])["yes"],
               "no": letter_ids(tok, ["No", "no"])["No"] + letter_ids(tok, ["no"])["no"]}

    if args.latency:
        rows = []
        cfgs = [("full", 576)] + [(m, k) for k in budgets for m in ("fastv", "cc4", "grid", "rand0")]
        sub = items[: args.latency]
        for ex in sub[:5]:  # warm-up
            for m, k in cfgs:
                R.timed_prefill(img_of(ex), ex["text"], m, k, ex["idx"])
        for m, k in cfgs:
            ts = [R.timed_prefill(img_of(ex), ex["text"], m, k, ex["idx"])[0] for ex in sub]
            torch.cuda.reset_peak_memory_stats()
            R.timed_prefill(img_of(sub[0]), sub[0]["text"], m, k, 0)
            peak = torch.cuda.max_memory_allocated() / 2**30
            rows.append(dict(method=m, k=k, n=len(ts), median_ms=1000 * float(np.median(ts)),
                             mean_ms=1000 * float(np.mean(ts)), p10_ms=1000 * float(np.percentile(ts, 10)),
                             p90_ms=1000 * float(np.percentile(ts, 90)), peak_mem_gib=peak))
            print(json.dumps(rows[-1]), flush=True)
        json.dump(dict(meta=meta, rows=rows), open(f"{args.out}/latency_{args.task}.json", "w"), indent=1)
        return

    # sanity: manual layer loop (no pruning) must reproduce HF's own forward
    for ex in items[: args.sanity]:
        out, _, _, emb, ids = R.prefill_all(img_of(ex), ex["text"], ["fastv"], [288], ex["idx"])
        with torch.no_grad():
            ref = R.model.language_model(inputs_embeds=emb).logits[0, -1].float() \
                if hasattr(R.model.language_model, "lm_head") else R.model(inputs_embeds=emb).logits[0, -1].float()
        d = (ref - out["full"]).abs().max().item()
        print(json.dumps(dict(sanity_qid=ex["qid"], max_abs_logit_diff=d,
                              same_argmax=bool(ref.argmax() == out["full"].argmax()))), flush=True)

    recs = []
    sel_counts = {f"{m}@{k}": np.zeros(N_IMG, dtype=np.int64) for k in budgets for m in methods}
    attn_sum = np.zeros(N_IMG, dtype=np.float64)
    t0 = time.time()
    for n, ex in enumerate(items):
        out, sels, a_last, emb, _ = R.prefill_all(img_of(ex), ex["text"], methods, budgets, ex["idx"])
        attn_sum += a_last / a_last.sum()
        for (m, k), s in sels.items():
            sel_counts[f"{m}@{k}"][s.cpu().numpy()] += 1
        rec = dict(idx=ex["idx"], qid=ex["qid"], answer=ex["answer"], seq_len=int(emb.shape[1]), prompt=ex["text"])
        if "category" in ex:
            rec["category"] = ex["category"]
        for name, lg in out.items():
            greedy = tok.decode([int(lg.argmax())]).strip()
            if args.task == "aokvqa":
                letters = "ABCD"[: ex["n_opt"]]
                sc = {Lt: max(lg[i].item() for i in lid[Lt]) for Lt in letters}
                pred = max(sc, key=sc.get)
                rec[name] = dict(pred=pred, greedy=greedy, ok=pred == ex["answer"],
                                 ok_greedy=ex["answer"] in greedy)
            else:
                sy = max(lg[i].item() for i in lid["yes"])
                sn = max(lg[i].item() for i in lid["no"])
                pred = "yes" if sy > sn else "no"
                rec[name] = dict(pred=pred, greedy=greedy, ok=pred == ex["answer"], margin=sy - sn)
        recs.append(rec)
        if n % 50 == 0 or n == len(items) - 1:
            acc = {k2: float(np.mean([r[k2]["ok"] for r in recs])) for k2 in ["full"] + [f"fastv@{b}" for b in budgets] + [f"cc4@{b}" for b in budgets]}
            print(json.dumps(dict(n=n + 1, sec=round(time.time() - t0), acc=acc)), flush=True)
    with open(f"{args.out}/preds_{args.task}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    np.savez(f"{args.out}/selection_{args.task}.npz", attn_mean=attn_sum / len(items),
             **{k2.replace("@", "_at_"): v for k2, v in sel_counts.items()})
    meta["runtime_sec"] = round(time.time() - t0)
    json.dump(meta, open(f"{args.out}/meta_{args.task}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
