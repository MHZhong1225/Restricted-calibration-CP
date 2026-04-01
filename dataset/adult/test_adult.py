import pandas as pd
df = pd.read_csv("./adult.csv")
print("Total rows:", len(df))
print("Target dist:", df["income"].value_counts())
print("Race dist:", df["race"].value_counts())
print("Sex dist:", df["sex"].value_counts())
