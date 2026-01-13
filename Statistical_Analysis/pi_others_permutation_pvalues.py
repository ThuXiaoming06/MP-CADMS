import os
import numpy as np
import pandas as pd

FILE_PATH = "Multi_pi.xlsx"
N_PERM = 10000
SEED = 42

CLASS_NAMES = ["IgAκ","IgAλ","IgGκ","IgGλ","IgMκ","IgMλ","κ","λ"]
N_CLASS = 8

USE_ARGMAX_PRED = True

LABEL_STR2INT = {name: i for i, name in enumerate(CLASS_NAMES)}

def parse_label_series(s: pd.Series, colname: str) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.astype(int)

    s_str = s.astype(str).str.strip()
    mapped = s_str.map(LABEL_STR2INT)
    out = mapped.copy().fillna(num)

    if out.isna().any():
        unknown = s_str[out.isna()].unique()
        raise ValueError(
            f"{colname} 中存在无法识别的值：{unknown}。\n"
            f"请检查拼写/空格，或补充 LABEL_STR2INT。"
        )
    return out.astype(int)

def average_precision_binary(y_true01: np.ndarray, y_score: np.ndarray) -> float:
    """Binary Average Precision (AP), sklearn-compatible."""
    y_true01 = np.asarray(y_true01).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()
    P = int(y_true01.sum())
    if P == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true01[order]

    tp = np.cumsum(y_sorted == 1)
    k = np.arange(1, len(y_sorted) + 1)
    precision_at_k = tp / k
    ap = precision_at_k[y_sorted == 1].sum() / P
    return float(ap)

def macro_auprc(y_true: np.ndarray, prob: np.ndarray) -> float:
    """macro AUPRC over 8 classes: mean(one-vs-rest AP_k)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    prob = np.asarray(prob).astype(float)
    aps = []
    for k in range(N_CLASS):
        y01 = (y_true == k).astype(int)
        aps.append(average_precision_binary(y01, prob[:, k]))
    return float(np.mean(aps))

def macro_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    macro metrics over 8 classes using one-vs-rest per class:
      sensitivity = recall
      specificity
      accuracy
      precision
      F1
    Then mean across classes.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    N = len(y_true)

    sens_list, spec_list, acc_list, prec_list, f1_list = [], [], [], [], []

    for k in range(N_CLASS):
        t = (y_true == k)
        p = (y_pred == k)

        tp = int(np.sum(t & p))
        fn = int(np.sum(t & (~p)))
        fp = int(np.sum((~t) & p))
        tn = int(np.sum((~t) & (~p)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc  = (tp + tn) / N if N > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        sens_list.append(sens)
        spec_list.append(spec)
        acc_list.append(acc)
        prec_list.append(prec)
        f1_list.append(f1)

    return {
        "sensitivity_macro": float(np.mean(sens_list)),
        "specificity_macro": float(np.mean(spec_list)),
        "accuracy_macro": float(np.mean(acc_list)),
        "precision_macro": float(np.mean(prec_list)),
        "F1-score_macro": float(np.mean(f1_list)),
    }

def compute_metrics(y_true: np.ndarray, prob: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {"AUPRC_macro": macro_auprc(y_true, prob)}
    out.update(macro_confusion_metrics(y_true, y_pred))
    return out

def permutation_test_two_groups(
    y_true: np.ndarray,
    prob: np.ndarray,
    y_pred: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
):
    """
    Two-sided permutation test for diff = metric(A) - metric(B)
    by shuffling group membership (keeping group sizes fixed).

    p = (#{|diff_perm| >= |diff_obs|} + 1) / (n_perm + 1)
    """
    rng = np.random.default_rng(seed)
    metric_names = [
        "AUPRC_macro",
        "sensitivity_macro",
        "specificity_macro",
        "accuracy_macro",
        "F1-score_macro",
        "precision_macro",
    ]

    idx_a = np.asarray(idx_a, dtype=int)
    idx_b = np.asarray(idx_b, dtype=int)
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError("某个分组样本数为0，无法进行置换检验。")

    def met(idx):
        return compute_metrics(y_true[idx], prob[idx, :], y_pred[idx])

    obs_a = met(idx_a)
    obs_b = met(idx_b)

    diff_obs = {m: obs_a[m] - obs_b[m] for m in metric_names}
    abs_diff_obs = {m: abs(diff_obs[m]) for m in metric_names}
    extreme = {m: 0 for m in metric_names}

    pooled = np.concatenate([idx_a, idx_b])
    n_a = len(idx_a)

    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        pa = perm[:n_a]
        pb = perm[n_a:]

        ma = met(pa)
        mb = met(pb)

        for m in metric_names:
            if abs(ma[m] - mb[m]) >= abs_diff_obs[m]:
                extreme[m] += 1

    pvals = {m: (extreme[m] + 1) / (n_perm + 1) for m in metric_names}
    return obs_a, obs_b, diff_obs, pvals

def load_multi_pi(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] < 13:
        raise ValueError("Multi_pi 列数不足，期望至少13列。")

    number_col = df.columns[0]
    prob_cols = list(df.columns[1:9])
    true_col = df.columns[9]
    pred_col = df.columns[10]
    gender_col = df.columns[11]
    age_col = df.columns[12]

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

    bad_y = set(np.unique(out["True_label"])) - set(range(N_CLASS))
    if bad_y:
        raise ValueError(f"True_label 出现不在0~7的值：{sorted(list(bad_y))}。请检查编码。")
    bad_g = set(np.unique(out["Gender"])) - {0, 1}
    if bad_g:
        raise ValueError(f"Gender 出现不在0/1的值：{sorted(list(bad_g))}。请检查编码。")

    if not USE_ARGMAX_PRED:
        out["Pred_label"] = parse_label_series(out["Pred_label"], "Pred_label")
        bad_p = set(np.unique(out["Pred_label"])) - set(range(N_CLASS))
        if bad_p:
            raise ValueError(f"Pred_label 出现不在0~7的值：{sorted(list(bad_p))}。请检查。")

    return out.reset_index(drop=True)

def age_group(a: float) -> str:
    if a <= 50:
        return "<=50"
    elif a <= 65:
        return "50-65"
    else:
        return ">65"

def main():
    df = load_multi_pi(FILE_PATH)

    prob = df[CLASS_NAMES].to_numpy()
    y_true = df["True_label"].to_numpy()

    if USE_ARGMAX_PRED:
        y_pred = np.argmax(prob, axis=1).astype(int)
        pred_source = "argmax(prob)"
    else:
        y_pred = df["Pred_label"].to_numpy().astype(int)
        pred_source = "excel_pred_label"

    df["AgeGroup"] = df["Age"].apply(age_group)

    comparisons = [
        ("Age <=50", "Age 50-65",
         np.where(df["AgeGroup"] == "<=50")[0],
         np.where(df["AgeGroup"] == "50-65")[0]),
        ("Age <=50", "Age >65",
         np.where(df["AgeGroup"] == "<=50")[0],
         np.where(df["AgeGroup"] == ">65")[0]),
        ("Age 50-65", "Age >65",
         np.where(df["AgeGroup"] == "50-65")[0],
         np.where(df["AgeGroup"] == ">65")[0]),
        ("Male", "Female",
         np.where(df["Gender"] == 1)[0],
         np.where(df["Gender"] == 0)[0]),
    ]

    rows = []
    for g1, g2, idx1, idx2 in comparisons:
        print(f"\n=== Permutation test: {g1} vs {g2} (n_perm={N_PERM}, two-sided) ===")

        obs1, obs2, diff_obs, pvals = permutation_test_two_groups(
            y_true=y_true,
            prob=prob,
            y_pred=y_pred,
            idx_a=idx1,
            idx_b=idx2,
            n_perm=N_PERM,
            seed=SEED,
        )

        for metric in ["AUPRC_macro","sensitivity_macro","specificity_macro","accuracy_macro","F1-score_macro","precision_macro"]:
            rows.append({
                "Comparison": f"{g1} vs {g2}",
                "Metric": metric,
                "N_group1": len(idx1),
                "N_group2": len(idx2),
                "Group1_value": obs1[metric],
                "Group2_value": obs2[metric],
                "Diff(Group1-Group2)": diff_obs[metric],
                "p_value_perm_2sided": pvals[metric],
                "n_perm": N_PERM,
                "seed": SEED,
                "pred_source": pred_source,
                "macro_over": "8 classes"
            })

    res = pd.DataFrame(rows).sort_values(["Comparison", "p_value_perm_2sided", "Metric"])
    out_path = os.path.join(os.path.dirname(FILE_PATH) if os.path.dirname(FILE_PATH) else ".",
                            "Multi_pi_age_gender_permutation_pvalues_macroMetrics.xlsx")
    res.to_excel(out_path, index=False)

    print("\n=== Done ===")
    print(res.to_string(index=False))
    print(f"\n已保存：{out_path}")

if __name__ == "__main__":
    main()
