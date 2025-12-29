import pandas as pd
import sys

try:
    df = pd.read_csv('data/output/batches.csv')
    if df.empty:
        print("No batches found.")
        sys.exit(0)

    # Group by day and sum quantity
    daily_prod = df.groupby('day_produced')['quantity'].sum().sort_index()
    
    print("\nDaily Production (Cases):")
    print(daily_prod.to_string())
    
    # Check for trend
    first_half = daily_prod.iloc[: len(daily_prod)//2].mean()
    second_half = daily_prod.iloc[len(daily_prod)//2 :].mean()
    
    print(f"\nAvg Production (First Half): {first_half:,.0f}")
    print(f"Avg Production (Second Half): {second_half:,.0f}")
    
    if second_half < first_half * 0.8:
        print("ALERT: Production dropping significantly!")
    else:
        print("Production looks stable.")

except Exception as e:
    print(f"Error: {e}")