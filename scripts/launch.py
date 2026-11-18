# Usage (on the VM, from the repo dir): python scripts/launch.py "<shell command>" <logfile>
import subprocess, sys
subprocess.Popen(f"nohup bash -c '{sys.argv[1]}' > {sys.argv[2]} 2>&1 &", shell=True)
print("launched:", sys.argv[1])
