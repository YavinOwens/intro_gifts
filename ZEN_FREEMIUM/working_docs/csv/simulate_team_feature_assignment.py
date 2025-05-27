import pandas as pd
from pathlib import Path

# Load data
base = Path(__file__).parent
product_df = pd.read_csv(base / 'Product Tracker - Product.csv', dtype=str)
team_df = pd.read_csv(base / 'Product Tracker - Team.csv', dtype=str)

# Standardize department names for matching
product_df['Department_std'] = product_df['Department'].str.replace(' ', '').str.lower()
team_df['Department_std'] = team_df['Department'].str.replace(' ', '').str.lower()

rows = []
for _, prod in product_df.iterrows():
    dept_std = prod['Department_std']
    team = prod['Team'].strip()
    product = prod['Product'].strip()
    features = float(prod['Features']) if pd.notna(prod['Features']) else 0
    # Get team members for this standardized department
    members = team_df[team_df['Department_std'] == dept_std]['Names'].tolist()
    n = len(members)
    if n == 0 or features == 0:
        continue
    # Example distribution: [2, 1, 0.5, 1, 0.5] for 5 features, 5 people
    # Generalize: assign 2 to first, 1 to second, 0.5 to third, 1 to fourth, 0.5 to fifth, repeat if more features/people
    base_dist = [2, 1, 0.5, 1, 0.5]
    dist = (base_dist * ((n // len(base_dist)) + 1))[:n]
    total = sum(dist)
    # Scale to match total features
    scale = features / total if total > 0 else 0
    dist = [round(x * scale, 2) for x in dist]
    for member, feat_count in zip(members, dist):
        rows.append({
            'Department': prod['Department'],
            'Team': team,
            'Product': product,
            'Team Member': member,
            'Features Assigned': feat_count
        })

sim_df = pd.DataFrame(rows)
out_path = base / 'Product Tracker - Team_Feature_Assignment_simulated.csv'
sim_df.to_csv(out_path, index=False)
print(f"Simulated assignment saved to {out_path}") 