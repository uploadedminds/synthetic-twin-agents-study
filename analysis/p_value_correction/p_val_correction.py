import os
import json
import numpy as np
from statsmodels.stats.multitest import multipletests


folder = "."

fdr_folder = os.path.join(folder, "fdr_adjusted")
os.makedirs(fdr_folder, exist_ok=True)

filename = "all_pvals.json"
filepath = os.path.join(folder, filename)

if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
    exit(1)

with open(filepath, "r") as f:
    data = json.load(f)

keys = list(data.keys())
arrays = [np.asarray(data[k], dtype=float) for k in keys]
lengths = [len(a) for a in arrays]

all_pvals = np.concatenate(arrays) if arrays else np.array([], dtype=float)

if all_pvals.size == 0:
    print(f"Skipping empty: {filename}")
    exit(1)


rejected, pvals_adj_global, pvals_correctedndarray, _ = multipletests(all_pvals, method="fdr_bh", alpha=0.05)

# Effective threshold = largest *raw* p-value that is still significant after FDR
if rejected.any():
    effective_threshold_raw = float(all_pvals[rejected].max())
    min_adj_sig = float(pvals_adj_global[rejected].min())
    max_adj_sig = float(pvals_adj_global[rejected].max())
    n_sig = int(rejected.sum())
else:
    effective_threshold_raw = None
    min_adj_sig = None
    max_adj_sig = None
    n_sig = 0

adjusted_data = {}
start = 0
for k, n in zip(keys, lengths):
    adjusted_data[k] = pvals_adj_global[start:start+n].tolist()
    start += n

out_adj = os.path.join(fdr_folder, filename.replace(".json", "_fdr.json"))
with open(out_adj, "w") as f:
    json.dump(adjusted_data, f, indent=4)
print(f"Adjusted p-values written: {out_adj}")

summary = {
    "alpha": 0.05,
    "n_tests": int(all_pvals.size),
    "n_significant": n_sig,
    "effective_fdr_threshold_raw": effective_threshold_raw,  # the "p < X (FDR-adjusted)" cutoff
    "min_adjusted_p_among_significant": min_adj_sig,
    "max_adjusted_p_among_significant": max_adj_sig,
}

out_sum = os.path.join(fdr_folder, filename.replace(".json", "_fdr_summary.json"))
with open(out_sum, "w") as f:
    json.dump(summary, f, indent=4)
print(f"Summary written: {out_sum}")
