
import csv
from collections import defaultdict

def analyze_production():
    daily_prod = defaultdict(float)
    
    with open('data/output/batches.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            day = int(row['day_produced'])
            qty = float(row['quantity'])
            daily_prod[day] += qty
            
    # Print typical daily production
    print("Day | Total Production (Cases)")
    print("-" * 30)
    for day in range(1, 366):
        print(f"{day:03d} | {daily_prod[day]:,.0f}")

if __name__ == "__main__":
    analyze_production()
