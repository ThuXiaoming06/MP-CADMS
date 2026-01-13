import pandas as pd
import numpy as np
from itertools import combinations

# ===== Configurate file name =====
files = {
    "D1": "D1.xlsx",
    "D2": "D2.xlsx",
    "D3": "D3.xlsx",
    "D4": "D4.xlsx",
    "D5": "D5.xlsx",
}

# Normalize PaID
def norm_paid(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace("\u3000", " ")
    if s == "":
        return None
    try:
        f = float(s)
        if f.is_integer():
            s = str(int(f))
    except:
        pass
    return s

# Read
dfs = {}
paid_sets = {}
for fold, path in files.items():
    df = pd.read_excel(path)
    if "PaID" not in df.columns:
        raise ValueError(f"{path} 中缺少 'PaID' 列")
    df["_PaID_norm"] = df["PaID"].apply(norm_paid)
    dfs[fold] = df
    paid_sets[fold] = set(df["_PaID_norm"].dropna().tolist())

# ===== 1. Check the intersections pair by pair =====
print("=== Pairwise PaID intersections ===")
any_overlap = False
for a, b in combinations(files.keys(), 2):
    inter = paid_sets[a] & paid_sets[b]
    if inter:
        any_overlap = True
        print(f"{a} ∩ {b}: {len(inter)} overlap(s) -> {sorted(list(inter))[:20]}{' ...' if len(inter)>20 else ''}")
    else:
        print(f"{a} ∩ {b}: 0")

# ===== 2. Global check: Which PaID appears in multiple datasets =====
from collections import defaultdict
owner = defaultdict(set)
for fold, s in paid_sets.items():
    for pid in s:
        owner[pid].add(fold)

dupes = {pid: sorted(list(folds)) for pid, folds in owner.items() if len(folds) > 1}

print("\n=== Global overlap summary ===")
if not dupes:
    print("✅ 没有发现跨数据集重复的 PaID。")
else:
    print(f"⚠️ 发现 {len(dupes)} 个 PaID 同时出现在多个数据集：")
    # 列出前若干
    preview = list(dupes.items())[:50]
    for pid, folds in preview:
        print(f"  PaID={pid} -> {folds}")
    if len(dupes) > 50:
        print("  ...（仅展示前50个）")

    # Generate a detailed Excel report
    dup_list = sorted(dupes.keys())

    detail_frames = []
    for fold, df in dfs.items():
        sub = df[df["_PaID_norm"].isin(dup_list)].copy()
        if not sub.empty:
            sub.insert(0, "Fold_oth", fold)
            detail_frames.append(sub)
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    # Summary table: The number of times each repeated PaID appears in each discount
    summary_rows = []
    for pid in dup_list:
        row = {"PaID": pid}
        for fold in files.keys():
            cnt = int((dfs[fold]["_PaID_norm"] == pid).sum())
            row[fold] = cnt
        row["Folds_present"] = ",".join(dupes[pid])
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("PaID")

    with pd.ExcelWriter("PaID_overlap_report.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary_by_PaID")
        if not detail.empty:
            detail.drop(columns=["_PaID_norm"], errors="ignore").to_excel(writer, index=False, sheet_name="detail_rows")
    print("已输出重复 PaID 报告：PaID_overlap_report.xlsx")

# ===== 3. Brief statistics =====
print("\n=== Fold-level PaID stats ===")
for fold in files.keys():
    print(f"{fold}: unique PaID={len(paid_sets[fold])}, rows={len(dfs[fold])}")
