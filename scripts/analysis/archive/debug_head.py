
import os
import subprocess

path = "data/output/standard_run/inventory.csv"
print(f"File exists: {os.path.exists(path)}")
print(f"File size: {os.path.getsize(path)}")

cmd = f"head -n 5 {path}"
print(f"Running: {cmd}")
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = proc.communicate()
print(f"STDOUT: {out.decode('utf-8')}")
print(f"STDERR: {err.decode('utf-8')}")
