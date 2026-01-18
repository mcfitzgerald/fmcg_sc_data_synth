
import pandas as pd
import os
import subprocess
from io import StringIO

def analyze_store_inventory(output_dir):
    print("Analyzing Store-Level Mix...")
    inv_path = os.path.join(output_dir, "inventory.csv")
    
    # 1. Get All SKUs (Reference)
    print("Getting Master SKU List...")
    cmd = f"grep 'SKU-' {inv_path} | cut -d, -f3 | head -n 10000 | sort | uniq"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate()
    master_skus = set(stdout.decode('utf-8').strip().split('\n'))
    print(f"Master SKU Count: {len(master_skus)}")
    
    # 2. Get Store SKUs
    target_store = "STORE-RET-001-0001"
    print(f"Scanning inventory for {target_store}...")
    cmd = f"tail -n 10000000 {inv_path} | grep '{target_store}'"
    
    try:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        stdout, _ = proc.communicate()
        header = "day,node_id,product_id,perceived_inventory,actual_inventory"
        df = pd.read_csv(StringIO(header + '\n' + stdout.decode('utf-8')))
        last_day = df['day'].max()
        df = df[df['day'] == last_day]
        
        store_skus = set(df['product_id'].unique())
        missing_skus = list(master_skus - store_skus)
        
        print(f"Store has {len(store_skus)} SKUs. Missing: {len(missing_skus)}")
        
        if missing_skus:
            targets = missing_skus[:5]
            print(f"Checking 5 Missing SKUs: {targets}")
            
            # Check RDC inventory for these
            # Assuming RDC-NE serves this store (based on ID, Northeast?)
            # Or RDC-MW. Let's check RDC-MW (Chicago) as it's central.
            target_rdc = "RDC-MW"
            print(f"Checking {target_rdc} inventory for missing items...")
            
            cmd = f"tail -n 10000000 {inv_path} | grep '{target_rdc}'"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            stdout, _ = proc.communicate()
            df_rdc = pd.read_csv(StringIO(header + '\n' + stdout.decode('utf-8')))
            df_rdc = df_rdc[df_rdc['day'] == last_day]
            
            print(f"\n--- RDC Inventory for Missing Store Items ---")
            for sku in targets:
                row = df_rdc[df_rdc['product_id'] == sku]
                if not row.empty:
                    qty = row['actual_inventory'].values[0]
                    print(f"{sku}: {qty:,.0f} cases")
                else:
                    print(f"{sku}: NOT FOUND AT RDC")

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    analyze_store_inventory("data/output/standard_run")
