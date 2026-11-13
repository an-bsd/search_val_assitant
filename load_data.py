import pandas as pd
import sqlite3
import zipfile
import glob
import os

# 1. Find the latest ZIP file
zip_path = max(glob.glob("faostat_sua_dataset/*.zip"), key=os.path.getmtime)

# 2. Read the FAOSTAT dataset
with zipfile.ZipFile(zip_path) as z:
    csv_name = next(name for name in z.namelist() if name.endswith(".csv") and "Normalized" in name)
    df = pd.read_csv(z.open(csv_name), low_memory=False)

# 3. Save to SQLite
conn = sqlite3.connect("data/faostat_sua.sqlite")

df.to_sql("sua", conn, if_exists="replace", index=False)

conn.close()