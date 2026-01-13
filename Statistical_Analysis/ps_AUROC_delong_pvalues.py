import os
import numpy as np
import pandas as pd
from scipy.stats import norm

FILE_PATH = "Multi_ps.xlsx"

CLASS_NAMES = ["WP", "P(+)", "SP(++)"]
N_CLASS = 3

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
        mid = 0.5 * (i + j - 1) + 1
        T[i:j] = mid
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out

def auc_and_delong_var(y_true01: np.ndarray, scores: np.ndarray):
    """
    单组样本：计算 AUC 及其 DeLong 方差估计（非参数）。
    y_true01: 0/1 (1=positive)
    scores: 连续得分（该类别概率）
    """
    y_true01 = np.asarray(y_true01).astype(int).ravel()
    scores = np.asarray(scores).astype(float).ravel()

    m = int(y_true01.sum())
    n = int((1 - y_true01).sum())
    if m == 0 or n == 0:
        raise ValueError("该组中必须同时有正例与负例，才能计算AUC/方差。")

    order = np.argsort(-y_true01)  # positives first
    s = scores[order]

    pos = s[:m]
    neg = s[m:]

    tz = _compute_midrank(s)
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)

    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)

    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m

    var_v01 = np.var(v01, ddof=1)
    var_v10 = np.var(v10, ddof=1)
    var_auc = var_v01 / m + var_v10 / n
    var_auc = max(var_auc, 1e-18)

    return float(auc), float(var_auc), m, n

def delong_independent_auc_test(y1, s1, y2, s2):
    """
    两独立组 AUC 差异检验（DeLong方差 + Z检验）:
      Z = (AUC1 - AUC2) / sqrt(var1 + var2)
      p = 2 * sf(|Z|)
    """
    auc1, var1, m1, n1 = auc_and_delong_var(y1, s1)
    auc2, var2, m2, n2 = auc_and_delong_var(y2, s2)

    delta = auc1 - auc2
    z = delta / np.sqrt(var1 + var2)
    p = 2 * norm.sf(abs(z))
    return auc1, auc2, float(delta), float(z), float(p), (m1, n1, m2, n2)

def stouffer_two_sided(pvals: np.ndarray, deltas: np.ndarray, weights: np.ndarray | None = None):
    """
    合并two-sided p-value，但保留方向：
      z_abs = norm.isf(p/2)
      z_signed = sign(delta) * z_abs
      Z = sum(w*z) / sqrt(sum(w^2))
      p_comb = 2*sf(|Z|)
    """
    pvals = np.asarray(pvals, dtype=float)
    deltas = np.asarray(deltas, dtype=float)

    pvals = np.clip(pvals, 1e-300, 1.0)
    z_abs = norm.isf(pvals / 2.0)
    z_signed = z_abs * np.sign(deltas)

    if weights is None:
        weights = np.ones_like(z_signed)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != z_signed.shape:
            raise ValueError("weights must match pvals shape")

    Z = np.sum(weights * z_signed) / np.sqrt(np.sum(weights ** 2))
    p_comb = 2 * norm.sf(abs(Z))
    return float(Z), float(p_comb)

def load_multi_ps(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] < 8:
        raise ValueError("Multi_ps 列数不足，期望至少8列。")

    number_col = df.columns[0]
    prob_cols = list(df.columns[1:4])
    true_col = df.columns[4]
    pred_col = df.columns[5]
    gender_col = df.columns[6]
    age_col = df.columns[7]

    out = df[[number_col] + prob_cols + [true_col, pred_col, gender_col, age_col]].copy()
    out.columns = ["Number"] + CLASS_NAMES + ["True_label", "Pred_label", "Gender", "Age"]

    out = out.dropna(subset=["True_label", "Gender", "Age"] + CLASS_NAMES)

    out["True_label"] = pd.to_numeric(out["True_label"], errors="coerce")
    out["Gender"] = pd.to_numeric(out["Gender"], errors="coerce")
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    for c in CLASS_NAMES:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["True_label", "Gender", "Age"] + CLASS_NAMES)

    out["True_label"] = out["True_label"].astype(int)
    out["Gender"] = out["Gender"].astype(int)

    bad_y = set(np.unique(out["True_label"])) - {0, 1, 2}
    if bad_y:
        raise ValueError(f"True_label 出现不在0~2的值：{sorted(list(bad_y))}。请检查编码。")
    bad_g = set(np.unique(out["Gender"])) - {0, 1}
    if bad_g:
        raise ValueError(f"Gender 出现不在0/1的值：{sorted(list(bad_g))}。请检查编码。")

    return out.reset_index(drop=True)

def age_group(a: float) -> str:
    if a <= 50:
        return "<=50"
    elif a <= 65:
        return "50-65"
    else:
        return ">65"

def run_comparison(df: pd.DataFrame, mask1: np.ndarray, mask2: np.ndarray, name1: str, name2: str):
    """
    对比两组：逐类DeLong得到p，再Stouffer合并。
    """
    d1 = df.loc[mask1].copy()
    d2 = df.loc[mask2].copy()

    if len(d1) == 0 or len(d2) == 0:
        raise ValueError(f"{name1} 或 {name2} 分组样本数为0，无法对比。")

    pvals = []
    deltas = []
    per_rows = []
    used = 0

    for k, cname in enumerate(CLASS_NAMES):
        y1 = (d1["True_label"].to_numpy() == k).astype(int)
        s1 = d1[cname].to_numpy()

        y2 = (d2["True_label"].to_numpy() == k).astype(int)
        s2 = d2[cname].to_numpy()

        if y1.sum() == 0 or y1.sum() == len(y1) or y2.sum() == 0 or y2.sum() == len(y2):
            per_rows.append({
                "Comparison": f"{name1} vs {name2}",
                "Class": cname,
                "Class_index": k,
                "N_group1": len(d1),
                "N_group2": len(d2),
                "AUROC_group1": np.nan,
                "AUROC_group2": np.nan,
                "Delta(AUC1-AUC2)": np.nan,
                "Z_stat": np.nan,
                "p_value_DeLong": np.nan,
                "Note": "Skipped (no pos or no neg in at least one group)"
            })
            continue

        auc1, auc2, delta, z, p, (m1, n1, m2, n2) = delong_independent_auc_test(y1, s1, y2, s2)

        per_rows.append({
            "Comparison": f"{name1} vs {name2}",
            "Class": cname,
            "Class_index": k,
            "N_group1": len(d1),
            "N_group2": len(d2),
            "Npos_group1(ovr)": int(m1),
            "Nneg_group1(ovr)": int(n1),
            "Npos_group2(ovr)": int(m2),
            "Nneg_group2(ovr)": int(n2),
            "AUROC_group1": auc1,
            "AUROC_group2": auc2,
            "Delta(AUC1-AUC2)": delta,
            "Z_stat": z,
            "p_value_DeLong": p,
            "Note": ""
        })

        pvals.append(p)
        deltas.append(delta)
        used += 1

    per_df = pd.DataFrame(per_rows)

    if used == 0:
        comb = {
            "Comparison": f"{name1} vs {name2}",
            "Classes_used": 0,
            "Z_Stouffer": np.nan,
            "p_value_Stouffer_2sided": np.nan
        }
    else:
        Zc, Pc = stouffer_two_sided(np.array(pvals), np.array(deltas), weights=None)
        comb = {
            "Comparison": f"{name1} vs {name2}",
            "Classes_used": used,
            "Z_Stouffer": Zc,
            "p_value_Stouffer_2sided": Pc
        }

    return per_df, comb


def main():
    df = load_multi_ps(FILE_PATH)
    df["AgeGroup"] = df["Age"].apply(age_group)

    comparisons = [
        ("<=50", "50-65",
         (df["AgeGroup"] == "<=50").to_numpy(),
         (df["AgeGroup"] == "50-65").to_numpy()),
        ("<=50", ">65",
         (df["AgeGroup"] == "<=50").to_numpy(),
         (df["AgeGroup"] == ">65").to_numpy()),
        ("50-65", ">65",
         (df["AgeGroup"] == "50-65").to_numpy(),
         (df["AgeGroup"] == ">65").to_numpy()),
        ("Male", "Female",
         (df["Gender"] == 1).to_numpy(),
         (df["Gender"] == 0).to_numpy()),
    ]

    all_per = []
    combined_rows = []

    for n1, n2, m1, m2 in comparisons:
        print(f"\n=== {n1} vs {n2}: per-class DeLong + Stouffer ===")
        per_df, comb = run_comparison(df, m1, m2, n1, n2)
        all_per.append(per_df)
        combined_rows.append(comb)
        print(f"  -> Stouffer: Z={comb['Z_Stouffer']:.4f}  p={comb['p_value_Stouffer_2sided']:.6g} (classes_used={comb['Classes_used']})")

    df_per = pd.concat(all_per, ignore_index=True)
    df_comb = pd.DataFrame(combined_rows).sort_values("p_value_Stouffer_2sided")

    out_path = os.path.join(os.path.dirname(FILE_PATH) if os.path.dirname(FILE_PATH) else ".",
                            "Multi_ps_age_gender_AUROC_DeLong_Stouffer.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_per.to_excel(writer, index=False, sheet_name="PerClass_DeLong")
        df_comb.to_excel(writer, index=False, sheet_name="Stouffer_Combined")

    print(f"\n已保存：{out_path}")

if __name__ == "__main__":
    main()
