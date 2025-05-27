import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Load product data
csv_path = Path(__file__).parent / 'Product Tracker - Product.csv'
df = pd.read_csv(csv_path, dtype=str)

# Prepare output
sim_rows = []

for _, row in df.iterrows():
    product = row['Product'].strip()
    features = int(row['Features']) if pd.notna(row['Features']) else 0
    dor = int(row['Features DOR']) if pd.notna(row['Features DOR']) else 0
    dod = int(row['Feteaures DOD']) if pd.notna(row['Feteaures DOD']) else 0
    backlog = int(row['Backlog']) if pd.notna(row['Backlog']) else 0
    release_date = pd.to_datetime(row['Feature Released'], dayfirst=True, errors='coerce')
    if pd.isna(release_date):
        continue
    # Simulate from 30 days before release to release
    start_date = release_date - timedelta(days=30)
    days = (release_date - start_date).days + 1
    for i in range(days):
        date = start_date + timedelta(days=i)
        # Linear simulation: features, dor, dod increase; backlog decreases
        feat_val = int(features * (i / days))
        dor_val = int(dor * (i / days))
        dod_val = int(dod * (i / days))
        backlog_val = max(backlog + features - feat_val, 0)
        sim_rows.append({
            'Product': product,
            'Date': date.strftime('%Y-%m-%d'),
            'Features': feat_val,
            'Features DOR': dor_val,
            'Feteaures DOD': dod_val,
            'Backlog': backlog_val
        })
    # Ensure last day matches final values
    sim_rows.append({
        'Product': product,
        'Date': release_date.strftime('%Y-%m-%d'),
        'Features': features,
        'Features DOR': dor,
        'Feteaures DOD': dod,
        'Backlog': backlog
    })

sim_df = pd.DataFrame(sim_rows)
out_path = Path(__file__).parent / 'Product Tracker - Product_simulation.csv'
sim_df.to_csv(out_path, index=False)
print(f"Simulation saved to {out_path}") 