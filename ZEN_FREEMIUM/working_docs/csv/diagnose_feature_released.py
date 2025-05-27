import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / 'Product Tracker - Product.csv'
df = pd.read_csv(csv_path, dtype=str)

print('--- COLUMN NAMES ---')
print(list(df.columns))

# Strip whitespace from headers and values
df.columns = [col.strip() for col in df.columns]
for col in df.columns:
    if col.lower().replace(' ', '') == 'featurereleased':
        df.rename(columns={col: 'Feature Released'}, inplace=True)

print('\n--- COLUMN NAMES (after strip/rename) ---')
print(list(df.columns))

if 'Feature Released' not in df.columns:
    print("ERROR: 'Feature Released' column not found after cleaning!")
    exit(1)

print('\n--- FEATURE RELEASED VALUES ---')
print(df['Feature Released'])
print('\nTYPES:')
print(df['Feature Released'].apply(type))

# Try parsing as date
dates = pd.to_datetime(df['Feature Released'], errors='coerce', dayfirst=True)
print('\n--- PARSED DATES ---')
print(dates)

missing = df[dates.isna()]
print(f"\nRows considered missing after parsing: {len(missing)}")
if not missing.empty:
    print(missing) 