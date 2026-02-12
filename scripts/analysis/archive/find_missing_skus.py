
import pandas as pd
import os
import subprocess
from io import StringIO
import re

def analyze_missing(output_dir):
    print("Loading data...")
    inv_path = os.path.join(output_dir, "inventory.csv")
    
    # We need to find ALL unique product_ids in the inventory file
    # Grep is faster than pandas for this
    print("Scanning for unique products...")
    # Grep all SKU- entries, cut column 3, sort unique
    cmd = f"grep 'SKU-' {inv_path} | cut -d, -f3 | sort | uniq"
    
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate()
    
    found_skus = set(stdout.decode('utf-8').strip().split('\n'))
    # remove empty strings
    found_skus = {s for s in found_skus if s}
    
    print(f"Total Unique Finished Goods Found: {len(found_skus)}")
    
    if len(found_skus) < 500:
        print("CRITICAL: Missing SKUs detected.")
        
        # Infer expected list
        # We know we have categories ORAL, PERSONAL, HOME
        # And IDs are usually sequential?
        # Let's just list the counts by prefix
        
        prefixes = {}
        for s in found_skus:
            p = s.split('-')[1] # SKU-ORAL-001 -> ORAL
            prefixes[p] = prefixes.get(p, 0) + 1
            
        print("Counts by Category:")
        for p, c in prefixes.items():
            print(f"  {p}: {c}")
            
        # Try to find holes
        for cat in prefixes:
            ids = []
            for s in found_skus:
                if f"SKU-{cat}" in s:
                    m = re.search(r'(\d+)$', s)
                    if m: ids.append(int(m.group(1)))
            
            ids.sort()
            full = set(range(1, max(ids)+1))
            actual = set(ids)
            missing = sorted(list(full - actual))
            if missing:
                print(f"  Missing IDs in {cat}: {missing[:10]} ... (Total {len(missing)})")

if __name__ == "__main__":
    analyze_missing("data/output/standard_run")
