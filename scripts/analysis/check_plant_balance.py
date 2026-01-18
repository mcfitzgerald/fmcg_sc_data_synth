
import pandas as pd
import os

def analyze_plant_load(output_dir):
    print("Loading batch history...")
    df = pd.read_csv(os.path.join(output_dir, "batches.csv"))
    
    print("\n=== TOTAL BATCHES PER PLANT ===")
    print(df['plant_id'].value_counts())
    
    print("\n=== TOTAL VOLUME PER PLANT (Cases) ===")
    vol = df.groupby('plant_id')['quantity'].sum()
    print(vol.apply(lambda x: f"{x:,.0f}"))
    
    # Check if any plant is consistently idle
    # Group by day and plant
    daily = df.groupby(['day_produced', 'plant_id']).size().unstack(fill_value=0)
    
    print("\n=== AVERAGE BATCHES PER DAY ===")
    print(daily.mean())
    
    print("\n=== DAYS WITH ZERO PRODUCTION ===")
    zero_days = (daily == 0).sum()
    print(zero_days)

if __name__ == "__main__":
    analyze_plant_load("data/output/standard_run")
