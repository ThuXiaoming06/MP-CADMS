import pandas as pd
import numpy as np

# ========== Adjustable parameters ==========
INPUT_FILE = "all_sample_information.xlsx"
OUTPUT_PREFIX = ""
SHEET_NAME = 0
RANDOM_SEED = 42

# Cost weight
W_PL1   = 2.0
W_PI1   = 2.0
W_UL0   = 5.0
W_SIZE  = 3.0

# Local rebalancing parameters
MAX_SWAPS = 400
IMPROVE_THRESHOLD = 1e-9

# A set of categories participating in equilibrium
PL_CATS = [0, 1, 2, 8]
PI_CATS = list(range(0, 9))  # 0..8

# ========== Reading and pre-checking ==========
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

required_cols = [
    "Number","Tiaoma","Label_PL","Label_PI","Label_Unlabel","CoTime","PaID","Gender","Age",
    "sFLC-κ","sFLC-λ","sFLC-κ/λ","U-Pro","24hUPr","α1","α2","Alb%","β1","β2","γ","A/G",
    "F-κ","F-λ","M Pro.","24hU-V","HGB","Ca","Cr(E)","CRP/hsCRP","CK","NT-proBNP","UA","LD",
    "Alb","PT","PT%","INR","Fbg","APTT","APTT-R","TT","D-Dimer","β2MG"
]
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Excel缺少必要列：{missing}")

for col in ["Label_PL","Label_PI","Label_Unlabel"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ========== Grouping (use "PaID" as the group; for missing items, use a unique line number for padding) ==========
group_id = df["PaID"].astype(object)
group_id = group_id.replace(r"^\s*$", np.nan, regex=True)
row_ids = pd.Series([f"ROW_{i}" for i in df.index], index=df.index)
group_id = group_id.where(group_id.notna(), row_ids)
df["_GroupID_"] = group_id

# ========== Global statistics ==========
def count_cats(series, cats):
    vc = series.value_counts(dropna=True)
    return np.array([vc.get(c, 0) for c in cats], dtype=float)

mask_ul1 = (df["Label_Unlabel"] == 1)
global_PL1_counts = count_cats(df.loc[mask_ul1, "Label_PL"], PL_CATS)
global_PI1_counts = count_cats(df.loc[mask_ul1, "Label_PI"], PI_CATS)

total_ul1 = int(mask_ul1.sum())
total_ul0 = int((df["Label_Unlabel"] == 0).sum())
total_samples = len(df)

PL1_target_each = global_PL1_counts / 5.0
PI1_target_each = global_PI1_counts / 5.0
UL0_target_each_4 = total_ul0 / 4.0
SIZE_TARGET = total_samples / 5.0

PL1_denom = np.maximum(PL1_target_each, 1.0)
PI1_denom = np.maximum(PI1_target_each, 1.0)

# ========== Group characteristics ==========
groups = []
for gid, sub in df.groupby("_GroupID_", sort=False):
    idxs = sub.index.to_list()
    n = len(sub)
    ul0 = int((sub["Label_Unlabel"] == 0).sum())
    sub_ul1 = sub[sub["Label_Unlabel"] == 1]
    pl1_cnt = count_cats(sub_ul1["Label_PL"], PL_CATS)
    pi1_cnt = count_cats(sub_ul1["Label_PI"], PI_CATS)
    ul1 = int(len(sub_ul1))
    groups.append({
        "gid": gid,
        "rows": idxs,
        "n": n,
        "ul0": ul0,
        "ul1": ul1,
        "pl1": pl1_cnt,
        "pi1": pi1_cnt
    })

# ========== Folding container ==========
FOLDS = ["D1","D2","D3","D4","D5"]
folds = {f: {
    "name": f, "rows": [], "n": 0, "ul0": 0, "ul1": 0,
    "pl1": np.zeros_like(global_PL1_counts),
    "pi1": np.zeros_like(global_PI1_counts)
} for f in FOLDS}

def add_group_to_fold(fold, g):
    fold["rows"].extend(g["rows"])
    fold["n"]   += g["n"]
    fold["ul0"] += g["ul0"]
    fold["ul1"] += g["ul1"]
    fold["pl1"] += g["pl1"]
    fold["pi1"] += g["pi1"]

def remove_group_from_fold(fold, g):
    for r in g["rows"]:
        fold["rows"].remove(r)
    fold["n"]   -= g["n"]
    fold["ul0"] -= g["ul0"]
    fold["ul1"] -= g["ul1"]
    fold["pl1"] -= g["pl1"]
    fold["pi1"] -= g["pi1"]

def fold_cost_components(fold):
    pl_cost = np.sum(((fold["pl1"] - PL1_target_each) / PL1_denom) ** 2) if total_ul1 > 0 else 0.0
    pi_cost = np.sum(((fold["pi1"] - PI1_target_each) / PI1_denom) ** 2) if total_ul1 > 0 else 0.0
    if fold["name"] != "D5":
        ul0_cost = abs(fold["ul0"] - UL0_target_each_4) / max(1.0, UL0_target_each_4)
    else:
        ul0_cost = 0.0
    size_cost = abs(fold["n"] - SIZE_TARGET) / max(1.0, SIZE_TARGET)
    return W_PL1*pl_cost + W_PI1*pi_cost + W_UL0*ul0_cost + W_SIZE*size_cost

def total_cost():
    return sum(fold_cost_components(folds[f]) for f in FOLDS)

def delta_cost_if_add(fold_name, g):
    f = folds[fold_name]
    if fold_name == "D5" and g["ul0"] > 0:
        return np.inf
    before = fold_cost_components(f)
    add_group_to_fold(f, g)
    after = fold_cost_components(f)
    remove_group_from_fold(f, g)
    return after - before

# ========== Initialization: Ensure that each fold has a seed ==========
rng = np.random.default_rng(RANDOM_SEED)

# 1) D5 first place a group of seeds without UL0 (if any)
candidates_d5 = [g for g in groups if g["ul0"] == 0]
if len(candidates_d5) > 0:
    seed_d5 = max(candidates_d5, key=lambda g: (g["ul1"], g["n"]))
    add_group_to_fold(folds["D5"], seed_d5)
    groups.remove(seed_d5)

# 2) place a group containing UL0 in D1 to D4 (if not enough, use a group without UL0 to make up for it)
candidates_ul0 = [g for g in groups if g["ul0"] > 0]
candidates_ul0.sort(key=lambda g: (g["ul0"], g["ul1"], g["n"]), reverse=True)
for fname in ["D1","D2","D3","D4"]:
    if candidates_ul0:
        g = candidates_ul0.pop(0)
        add_group_to_fold(folds[fname], g)
        groups.remove(g)
for fname in ["D1","D2","D3","D4"]:
    if folds[fname]["n"] == 0 and groups:
        g = max(groups, key=lambda x: (x["ul1"], x["n"]))
        add_group_to_fold(folds[fname], g)
        groups.remove(g)

# ========== Principal greed: Place the remaining groups at incremental cost ==========
groups.sort(key=lambda g: (g["ul0"], g["ul1"], g["n"]), reverse=True)
for g in groups:
    best_f, best_dc = None, np.inf
    for fname in FOLDS:
        dc = delta_cost_if_add(fname, g)
        if dc < best_dc:
            best_f, best_dc = fname, dc
    add_group_to_fold(folds[best_f], g)

# ========== Local rebalancing: Try to relocate between the most full and the most empty folds ==========
def group_stats_from_gid(gid):
    sub = df[df["_GroupID_"] == gid]
    n = len(sub)
    ul0 = int((sub["Label_Unlabel"] == 0).sum())
    sub_ul1 = sub[sub["Label_Unlabel"] == 1]
    pl1_cnt = count_cats(sub_ul1["Label_PL"], PL_CATS)
    pi1_cnt = count_cats(sub_ul1["Label_PI"], PI_CATS)
    ul1 = int(len(sub_ul1))
    return {"gid": gid, "rows": sub.index.tolist(), "n": n, "ul0": ul0, "ul1": ul1,
            "pl1": pl1_cnt, "pi1": pi1_cnt}

def try_rebalance_once():
    sizes = sorted([(fname, folds[fname]["n"]) for fname in FOLDS], key=lambda t: t[1])
    src_name = sizes[-1][0]
    dst_name = sizes[0][0]
    if src_name == dst_name:
        return False
    src, dst = folds[src_name], folds[dst_name]
    total_before = total_cost()

    # src 中有哪些 gid
    src_gids = df.loc[src["rows"], "_GroupID_"].unique().tolist()
    best_gain, best_group = 0.0, None

    for gid in src_gids:
        g = group_stats_from_gid(gid)
        if dst_name == "D5" and g["ul0"] > 0:
            continue
        remove_group_from_fold(src, g)
        add_group_to_fold(dst, g)
        after = total_cost()
        gain = total_before - after
        remove_group_from_fold(dst, g)
        add_group_to_fold(src, g)
        if gain > best_gain + IMPROVE_THRESHOLD:
            best_gain, best_group = gain, g

    if best_group is None:
        return False
    remove_group_from_fold(folds[src_name], best_group)
    add_group_to_fold(folds[dst_name], best_group)
    return True

swaps = 0
while swaps < MAX_SWAPS and try_rebalance_once():
    swaps += 1
# print(f"Local rebalancing swaps: {swaps}")

# ========== Write back the result ==========
assignment = {}
for f in FOLDS:
    for r in folds[f]["rows"]:
        assignment[r] = f
df["Fold"] = df.index.map(lambda i: assignment.get(i, None))

for f in FOLDS:
    sub = df[df["Fold"] == f].copy()
    out_path = f"{OUTPUT_PREFIX}{f}_v2.xlsx"
    sub.to_excel(out_path, index=False)
    print(f"{f}: {len(sub)} rows -> {out_path}")

df.to_excel(f"{OUTPUT_PREFIX}all_sample_information_with_folds_v2.xlsx", index=False)

# ========== Print the allocation result ==========
print("\n=== Global targets ===")
print(f"Total samples: {total_samples}  |  per-fold target ≈ {SIZE_TARGET:.1f}")
print(f"Total UL==0:   {total_ul0}      |  D1~D4 target each ≈ {UL0_target_each_4:.1f}")
print(f"Total UL==1:   {total_ul1}")
print(f"Global PL(UL==1) counts [0,1,2,8]: {global_PL1_counts.astype(int).tolist()}  "
      f"=> target each ≈ {np.round(PL1_target_each,1).tolist()}")
print(f"Global PI(UL==1) counts [0..8]:   {global_PI1_counts.astype(int).tolist()}  "
      f"=> target each ≈ {np.round(PI1_target_each,1).tolist()}")

print("\n=== Per-fold details (counts & deviation from target) ===")
for fname in FOLDS:
    fold = folds[fname]
    pl_counts = fold["pl1"].astype(int)
    pi_counts = fold["pi1"].astype(int)
    pl_diff = np.round(pl_counts - PL1_target_each, 1)
    pi_diff = np.round(pi_counts - PI1_target_each, 1)

    print(f"\n-- {fname} --")
    print(f"samples={fold['n']},  UL0={fold['ul0']},  UL1={fold['ul1']}")
    if fname != "D5":
        print(f"UL0 target≈{UL0_target_each_4:.1f},  diff={fold['ul0']-UL0_target_each_4:.1f}")
    print(f"PL(UL==1) counts [0,1,2,8]: {pl_counts.tolist()}  diff_from_target: {pl_diff.tolist()}")
    print(f"PI(UL==1) counts [0..8]:   {pi_counts.tolist()}  diff_from_target: {pi_diff.tolist()}")

# Constraint check
print("\n=== Constraint checks ===")
viol_d5_ul0 = int((df[(df["Fold"]=="D5") & (df["Label_Unlabel"]==0)]).shape[0])
print(f"D5 contains UL==0 samples?  {'YES -> '+str(viol_d5_ul0) if viol_d5_ul0>0 else 'NO'}")

ul0_counts_d1_4 = [int((df[(df['Fold']==f) & (df['Label_Unlabel']==0)]).shape[0]) for f in ["D1","D2","D3","D4"]]
print(f"D1~D4 UL==0 counts: {ul0_counts_d1_4}  "
      f"(target each≈{UL0_target_each_4:.1f},  max-min diff={max(ul0_counts_d1_4)-min(ul0_counts_d1_4) if ul0_counts_d1_4 else 0})")

sizes = [int(df[df['Fold']==f].shape[0]) for f in FOLDS]
print(f"Fold sizes: {dict(zip(FOLDS, sizes))}  (max-min diff={max(sizes)-min(sizes) if sizes else 0})")

# ummary of concise tables
def fold_summary_counts(fold_name):
    fold = folds[fold_name]
    return {
        "Fold": fold_name,
        "Samples": fold["n"],
        "UL0_count": fold["ul0"],
        "UL1_count": fold["ul1"],
        "PL(UL==1) counts [0,1,2,8]": fold["pl1"].astype(int).tolist(),
        "PI(UL==1) counts [0..8]":    fold["pi1"].astype(int).tolist()
    }

summary = pd.DataFrame([fold_summary_counts(f) for f in FOLDS])
print("\n=== Split Summary (table) ===")
print(summary.to_string(index=False))
