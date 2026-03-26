import pandas as pd
df = pd.read_csv("./nursery.csv")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
for c in df.columns:
    print(f"{c}: {df[c].nunique()} unique values")
print("Target dist:")
print(df["final evaluation"].value_counts())
