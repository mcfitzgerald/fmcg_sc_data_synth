
import pandas as pd
import json
import os
import subprocess
from io import StringIO

def analyze_inventory_distribution(output_dir):
    print("Loading inventory snapshot (tail)...")
    file_path = os.path.join(output_dir, "inventory.csv")
    
    # Get header
    with open(file_path, 'r') as f:
        header = f.readline()
        
    # Get last 500k lines
    proc = subprocess.Popen(['tail', '-n', '500000', file_path], stdout=subprocess.PIPE)
    stdout, _ = proc.communicate()
    data = stdout.decode('utf-8')
    
    # Combine
    csv_str = header + data
    
    try:
        df = pd.read_csv(StringIO(csv_str), on_bad_lines='skip') # Skip lines that might be cut off
        
        print(f"Analyzed {len(df)} rows from end of simulation.")
        print("Columns:", df.columns.tolist())
        
        def get_node_type(node_id):
            if not isinstance(node_id, str): return "UNKNOWN"
            if "STORE" in node_id: return "STORE"
            if "RDC" in node_id: return "RDC"
            if "PLANT" in node_id: return "PLANT"
            if "RET-DC" in node_id: return "RET_DC"
            if "DIST-DC" in node_id: return "DIST_DC"
            if "ECOM" in node_id: return "ECOM"
            if "SUP" in node_id: return "SUPPLIER"
            return "OTHER"

        df['node_type'] = df['node_id'].apply(get_node_type)
        
        # Total Inventory by Node Type
        print("\n--- Inventory by Node Type (End State) ---")
        print(df.groupby('node_type')['actual_inventory'].sum().apply(lambda x: f"{x:,.0f}"))

        # Check raw materials vs finished goods
        def get_category(prod_id):
            if not isinstance(prod_id, str): return "UNKNOWN"
            if prod_id.startswith("ING") or prod_id.startswith("PKG") or prod_id.startswith("ACT") or prod_id.startswith("BLK"):
                return "RAW_MAT"
            return "FINISHED_GOOD"

        df['category'] = df['product_id'].apply(get_category)
        
        print("\n--- Finished Goods vs Raw Materials ---")
        print(df.groupby(['category', 'node_type'])['actual_inventory'].sum().apply(lambda x: f"{x:,.0f}"))

    except Exception as e:
        print(f"Error reading inventory: {e}")

if __name__ == "__main__":
    analyze_inventory_distribution("data/output/standard_run")
