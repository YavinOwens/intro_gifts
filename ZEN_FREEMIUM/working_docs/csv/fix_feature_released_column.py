import pandas as pd
from pathlib import Path
import shutil

csv_path = Path(__file__).parent / 'Product Tracker - Product.csv'
backup_path = Path(__file__).parent / 'Product Tracker - Product_backup.csv'

df = pd.read_csv(csv_path, dtype=str)
# Clean headers
cols = [col.strip() for col in df.columns]
df.columns = cols

# Standardize column names
for col in df.columns:
    if col.lower().replace(' ', '') == 'featurereleased':
        df.rename(columns={col: 'Feature Released'}, inplace=True)
    if col.lower().replace(' ', '') == 'nextrelease':
        df.rename(columns={col: 'Next Release'}, inplace=True)

# Backup
shutil.copy(csv_path, backup_path)
print(f"Backup saved to {backup_path}")

# Move dates from 'Next Release' to 'Feature Released' if needed
if 'Feature Released' in df.columns and 'Next Release' in df.columns:
    # Only move if 'Feature Released' is not a valid date
    fr_dates = pd.to_datetime(df['Feature Released'], errors='coerce', dayfirst=True)
    nr_dates = pd.to_datetime(df['Next Release'], errors='coerce', dayfirst=True)
    to_fix = fr_dates.isna() & nr_dates.notna()
    print(f"Rows to fix: {to_fix.sum()}")
    df.loc[to_fix, 'Feature Released'] = df.loc[to_fix, 'Next Release']
    # Optionally clear 'Next Release' if you want
    # df.loc[to_fix, 'Next Release'] = ''
else:
    print("ERROR: Required columns not found.")

# Save
df.to_csv(csv_path, index=False)
print(f"Updated 'Feature Released' column in {csv_path}") 