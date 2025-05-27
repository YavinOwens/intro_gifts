import pandas as pd
from datetime import datetime
import shutil
from pathlib import Path

# File paths
csv_path = Path(__file__).parent / 'Product Tracker - Product.csv'
backup_path = Path(__file__).parent / 'Product Tracker - Product_backup.csv'

# If file does not exist, create it with headers
if not csv_path.exists():
    headers = ['Client', 'Files', 'Department', 'Team', 'Product', 'Features', 'Features DOR', 'Feteaures DOD', 'Backlog', 'Feature Released', 'Next Release', 'Status', 'Skill Required']
    pd.DataFrame(columns=headers).to_csv(csv_path, index=False)
    print(f"Created new file: {csv_path}")

# Load CSV
df = pd.read_csv(csv_path, dtype=str)

# Strip whitespace from headers
original_columns = df.columns.tolist()
df.columns = [col.strip() for col in df.columns]

# Standardize 'Feature Released' column name
for col in df.columns:
    if col.lower().replace(' ', '') == 'featurereleased':
        df.rename(columns={col: 'Feature Released'}, inplace=True)

# Ensure 'Feature Released' column exists
if 'Feature Released' not in df.columns:
    df['Feature Released'] = ''

# Backup original file
shutil.copy(csv_path, backup_path)
print(f"Backup saved to {backup_path}")

# Find missing dates
missing = df[df['Feature Released'].isna() | (df['Feature Released'].astype(str).str.strip() == '')]
print(f"Found {len(missing)} products missing 'Feature Released' date.")
if not missing.empty and set(['Product', 'Team', 'Department']).issubset(df.columns):
    print(missing[['Product', 'Team', 'Department']])

# Fill missing with today's date
fill_date = datetime.today().strftime('%Y-%m-%d')
df['Feature Released'] = df['Feature Released'].fillna('').astype(str)
df.loc[df['Feature Released'].str.strip() == '', 'Feature Released'] = fill_date

# Save updated CSV
df.to_csv(csv_path, index=False)
print(f"Filled missing 'Feature Released' dates with {fill_date} and saved to {csv_path}") 