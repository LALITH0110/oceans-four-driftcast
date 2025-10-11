import sys, pandas as pd
p = sys.argv[1]
df = pd.read_csv(p, skiprows=[1])
out = p.replace(".csv","_clean.csv")
df.to_csv(out, index=False)
print("Cleaned ->", out)
