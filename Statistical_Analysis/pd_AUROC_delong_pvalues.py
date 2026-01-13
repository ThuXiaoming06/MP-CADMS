import os
import numpy as np
import pandas as pd
from scipy.stats import norm

FILE_PATH = "Multi_np.xlsx"

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)

    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1  # 1-based
        T[i:j] = mid
        i = j

    out = np.empty(N, dtype=float)
    out[J] = T
    return out

def auc_and_delong_var(y_true01: np.ndarray, scores: np.ndarray):
    """
    For ONE set of samples (one model), compute:
      AUC and DeLong variance estimate.

    y_true01: 0/1 labels (1=positive)
    scores: continuous scores for positive class
    """
    y_true01 = np.asarray(y_true01).astype(int).ravel()
    scores = np.asarray(scores).astype(float).ravel()

    m = int(y_true01.sum())
    n = int((1 - y_true01).sum())
    if m == 0 or n == 0:
        raise ValueError("该分组内必须同时包含正例与负例，否则无法计算AUROC/方差。")

    # positives first
    order = np.argsort(-y_true01)
    s = scores[order]
    y = y_true01[order]

    # split
    pos = s[:m]
    neg = s[m:]

    tz = _compute_midrank(s)
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)

    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)

    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m

    # unbiased sample variances
    var_v01 = np.var(v01, ddof=1)
    var_v10 = np.var(v10, ddof=1)

    var_auc = var_v01 / m + var_v10 / n
    var_auc = max(var_auc, 1e-18)

    return float(auc), float(var_auc), m, n

def delong_independent_auc_test(y1, s1, y2, s2):
    """
    Compare AUC between two INDEPENDENT groups using DeLong variance estimates:
      Z = (AUC1 - AUC2) / sqrt(var1 + var2)
    Return: auc1, auc2, delta, z, p, (m1,n1,m2,n2)
    """
    auc1, var1, m1, n1 = auc_and_delong_var(y1, s1)
    auc2, var2, m2, n2 = auc_and_delong_var(y2, s2)

    delta = auc1 - auc2
    z = delta / np.sqrt(var1 + var2)
    p = 2 * norm.sf(abs(z))
    return auc1, auc2, float(delta), float(z), float(p), (m1, n1, m2, n2)

def load_multi_np(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] < 7:
        raise ValueError("Multi_np 列数不足，期望至少7列。")

    number_col = df.columns[0]
    pos_col = df.columns[2]
    true_col = df.columns[3]
    gender_col = df.columns[5]
    age_col = df.columns[6]

    out = df[[number_col, pos_col, true_col, gender_col, age_col]].copy()
    out.columns = ["Number", "Pos_prob", "True_label", "Gender", "Age"]

    out = out.dropna(subset=["Pos_prob", "True_label", "Gender", "Age"])

    out["Pos_prob"] = pd.to_numeric(out["Pos_prob"], errors="coerce")
    out["True_label"] = pd.to_numeric(out["True_label"], errors="coerce")
    out["Gender"] = pd.to_numeric(out["Gender"], errors="coerce")
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")

    out = out.dropna(subset=["Pos_prob", "True_label", "Gender", "Age"])

    out["True_label"] = out["True_label"].astype(int)
    if not set(np.unique(out["True_label"])) <= {0, 1}:
        raise ValueError("True_label 不是0/1二分类编码，请检查。")

    out["Gender"] = out["Gender"].astype(int)
    if not set(np.unique(out["Gender"])) <= {0, 1}:
        raise ValueError("Gender 不是0/1编码，请检查。")

    return out.reset_index(drop=True)

def age_group(age: float) -> str:
    if age <= 50:
        return "<=50"
    elif age <= 65:
        return "50-65"
    else:
        return ">65"

def main():
    df = load_multi_np(FILE_PATH)
    df["AgeGroup"] = df["Age"].apply(age_group)

    def get_group(sub_df: pd.DataFrame):
        y = sub_df["True_label"].to_numpy()
        s = sub_df["Pos_prob"].to_numpy()
        return y, s, len(sub_df)

    comparisons = [
        ("Age <=50", "Age 50-65",
         df[df["AgeGroup"] == "<=50"], df[df["AgeGroup"] == "50-65"]),
        ("Age <=50", "Age >65",
         df[df["AgeGroup"] == "<=50"], df[df["AgeGroup"] == ">65"]),
        ("Age 50-65", "Age >65",
         df[df["AgeGroup"] == "50-65"], df[df["AgeGroup"] == ">65"]),
        ("Male", "Female",
         df[df["Gender"] == 1], df[df["Gender"] == 0]),
    ]

    rows = []
    for g1_name, g2_name, d1, d2 in comparisons:
        y1, s1, n1 = get_group(d1)
        y2, s2, n2 = get_group(d2)

        print(f"\n=== DeLong (independent groups): {g1_name} vs {g2_name} ===")

        auc1, auc2, delta, z, p, (m1, nneg1, m2, nneg2) = delong_independent_auc_test(y1, s1, y2, s2)

        rows.append({
            "Comparison": f"{g1_name} vs {g2_name}",
            "N_group1": n1,
            "N_group2": n2,
            "Npos_group1": int(m1),
            "Nneg_group1": int(nneg1),
            "Npos_group2": int(m2),
            "Nneg_group2": int(nneg2),
            "AUROC_group1": auc1,
            "AUROC_group2": auc2,
            "Delta(AUC1-AUC2)": delta,
            "Z_stat": z,
            "p_value_DeLong_2sided": p
        })

        print(f"  AUC1={auc1:.4f}, AUC2={auc2:.4f}, delta={delta:.4f}, p={p:.6g}")

    res = pd.DataFrame(rows).sort_values("p_value_DeLong_2sided")
    out_path = os.path.join(os.path.dirname(FILE_PATH) if os.path.dirname(FILE_PATH) else ".", "Multi_np_age_gender_DeLong_AUROC_pvalues.xlsx")
    res.to_excel(out_path, index=False)

    print("\n=== Done ===")
    print(res.to_string(index=False))
    print(f"\n已保存：{out_path}")

if __name__ == "__main__":
    main()
