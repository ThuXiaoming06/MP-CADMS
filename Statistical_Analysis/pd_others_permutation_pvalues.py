import os
import numpy as np
import pandas as pd

FILE_PATH = "Multi_np.xlsx"
N_PERM = 10000
SEED = 42

USE_PROB_THRESHOLD_PRED = True
THRESHOLD = 0.5

def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary Average Precision (AUPRC / AP), sklearn-compatible."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()

    P = int(y_true.sum())
    if P == 0:
        return 0.0

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    k = np.arange(1, len(y_sorted) + 1)
    precision_at_k = tp / k

    ap = precision_at_k[y_sorted == 1].sum() / P
    return float(ap)

def confusion_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(acc),
        "precision": float(prec),
        "F1-score": float(f1),
    }

def compute_metrics(y_true: np.ndarray, pos_prob: np.ndarray, pred_label: np.ndarray) -> dict:
    out = {"AUPRC": average_precision_binary(y_true, pos_prob)}
    out.update(confusion_metrics_binary(y_true, pred_label))
    return out

def permutation_test_two_groups(
    y_true: np.ndarray,
    pos_prob: np.ndarray,
    pred_label: np.ndarray,
    group_mask_a: np.ndarray,
    group_mask_b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
):
    """
    Two-sided permutation test for metric difference between group A and group B.
    Null: group labels exchangeable.
    Statistic: diff = metric(A) - metric(B)
    p = (#{|diff_perm| >= |diff_obs|} + 1) / (n_perm + 1)
    """
    rng = np.random.default_rng(seed)

    metric_names = ["AUPRC", "sensitivity", "specificity", "accuracy", "F1-score", "precision"]

    idx_a = np.where(group_mask_a)[0]
    idx_b = np.where(group_mask_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError("某个分组样本数为0，无法做对比。")

    def metrics_on_indices(idx):
        return compute_metrics(y_true[idx], pos_prob[idx], pred_label[idx])

    obs_a = metrics_on_indices(idx_a)
    obs_b = metrics_on_indices(idx_b)
    diff_obs = {m: obs_a[m] - obs_b[m] for m in metric_names}
    abs_diff_obs = {m: abs(diff_obs[m]) for m in metric_names}
    extreme = {m: 0 for m in metric_names}

    # pooled indices
    pooled = np.concatenate([idx_a, idx_b])
    n_a = len(idx_a)

    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        pa = perm[:n_a]
        pb = perm[n_a:]

        met_a = metrics_on_indices(pa)
        met_b = metrics_on_indices(pb)

        for m in metric_names:
            if abs(met_a[m] - met_b[m]) >= abs_diff_obs[m]:
                extreme[m] += 1

    pvals = {m: (extreme[m] + 1) / (n_perm + 1) for m in metric_names}
    return obs_a, obs_b, diff_obs, pvals, len(idx_a), len(idx_b)

def load_multi_np(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] < 7:
        raise ValueError("Multi_np 列数不足，期望至少7列。")

    number_col = df.columns[0]
    neg_col = df.columns[1]
    pos_col = df.columns[2]
    true_col = df.columns[3]
    pred_col = df.columns[4]
    gender_col = df.columns[5]
    age_col = df.columns[6]

    out = df[[number_col, neg_col, pos_col, true_col, pred_col, gender_col, age_col]].copy()
    out.columns = ["Number", "Neg_prob", "Pos_prob", "True_label", "Pred_label", "Gender", "Age"]

    out = out.dropna(subset=["Number", "Pos_prob", "True_label", "Gender", "Age"])
    out["Pos_prob"] = pd.to_numeric(out["Pos_prob"], errors="coerce")
    out["True_label"] = pd.to_numeric(out["True_label"], errors="coerce")
    out["Pred_label"] = pd.to_numeric(out["Pred_label"], errors="coerce")
    out["Gender"] = pd.to_numeric(out["Gender"], errors="coerce")
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")

    out = out.dropna(subset=["Pos_prob", "True_label", "Gender", "Age"])

    out["True_label"] = out["True_label"].astype(int)
    if not set(np.unique(out["True_label"])) <= {0, 1}:
        raise ValueError("True_label 不是0/1二分类编码，请检查。")

    out["Gender"] = out["Gender"].astype(int)
    if not set(np.unique(out["Gender"])) <= {0, 1}:
        raise ValueError("Gender 不是0/1编码，请检查。")

    if not USE_PROB_THRESHOLD_PRED:
        out = out.dropna(subset=["Pred_label"])
        out["Pred_label"] = out["Pred_label"].astype(int)

    return out.reset_index(drop=True)

def make_age_group(age: float) -> str:
    if age <= 50:
        return "<=50"
    elif age <= 65:
        return "50-65"
    else:
        return ">65"

def main():
    df = load_multi_np(FILE_PATH)

    pos_prob = df["Pos_prob"].to_numpy()
    y_true = df["True_label"].to_numpy()

    if USE_PROB_THRESHOLD_PRED:
        y_pred = (pos_prob >= THRESHOLD).astype(int)
        pred_source = f"pos_prob>={THRESHOLD}"
    else:
        y_pred = df["Pred_label"].to_numpy().astype(int)
        pred_source = "excel_pred_label"

    age_groups = df["Age"].apply(make_age_group)
    gender = df["Gender"]

    comparisons = [
        ("Age <=50", "Age 50-65", (age_groups == "<=50").to_numpy(), (age_groups == "50-65").to_numpy()),
        ("Age <=50", "Age >65",   (age_groups == "<=50").to_numpy(), (age_groups == ">65").to_numpy()),
        ("Age 50-65","Age >65",   (age_groups == "50-65").to_numpy(), (age_groups == ">65").to_numpy()),
        ("Male",     "Female",    (gender == 1).to_numpy(), (gender == 0).to_numpy()),
    ]

    rows = []
    for g1_name, g2_name, m1, m2 in comparisons:
        print(f"\n=== Permutation test: {g1_name} vs {g2_name} (n_perm={N_PERM}, two-sided) ===")
        obs1, obs2, diff_obs, pvals, n1, n2 = permutation_test_two_groups(
            y_true=y_true,
            pos_prob=pos_prob,
            pred_label=y_pred,
            group_mask_a=m1,
            group_mask_b=m2,
            n_perm=N_PERM,
            seed=SEED
        )

        for metric in ["AUPRC", "sensitivity", "specificity", "accuracy", "F1-score", "precision"]:
            rows.append({
                "Comparison": f"{g1_name} vs {g2_name}",
                "Metric": metric,
                "N_group1": n1,
                "N_group2": n2,
                "Group1_value": obs1[metric],
                "Group2_value": obs2[metric],
                "Diff(Group1-Group2)": diff_obs[metric],
                "p_value_perm_2sided": pvals[metric],
                "n_perm": N_PERM,
                "seed": SEED,
                "pred_source": pred_source
            })

    res = pd.DataFrame(rows).sort_values(["Comparison", "p_value_perm_2sided", "Metric"])
    out_path = os.path.join(os.path.dirname(FILE_PATH) if os.path.dirname(FILE_PATH) else ".", "Multi_np_age_gender_permutation_pvalues.xlsx")
    res.to_excel(out_path, index=False)

    print("\n=== Done ===")
    print(res.to_string(index=False))
    print(f"\n已保存：{out_path}")

if __name__ == "__main__":
    main()
