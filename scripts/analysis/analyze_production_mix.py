
import pandas as pd
import os

def analyze_production(output_dir):
    print("Loading production batches...")
    df = pd.read_csv(os.path.join(output_dir, "batches.csv"))
    
    unique_products = df['product_id'].nunique()
    print(f"Total Unique Products Produced: {unique_products} (Target: ~500)")
    
    if unique_products == 0:
        print("CRITICAL: No production batches found!")
        return

    # Count batches per product
    counts = df['product_id'].value_counts()
    
    print("\n--- Top 10 Produced SKUs ---")
    print(counts.head(10))
    
    print("\n--- Bottom 10 Produced SKUs ---")
    print(counts.tail(10))
    
    # Check volume distribution
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    vol_by_sku = df.groupby('product_id')['quantity'].sum().sort_values(ascending=False)
    
    print("\n--- Volume Distribution (Top 5 vs Bottom 5) ---")
    print(vol_by_sku.head(5))
    print(vol_by_sku.tail(5))
    
    # Check if any production stopped?
    # Last production day per SKU
    last_day = df.groupby('product_id')['day_produced'].max().sort_values()
    print("\n--- Earliest 'Last Production Day' (Did some stop producing?) ---")
    print(last_day.head(10))

if __name__ == "__main__":
    analyze_production("data/output/standard_run")
