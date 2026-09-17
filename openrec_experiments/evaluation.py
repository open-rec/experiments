import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def binary_metrics(labels, scores):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid labels/prediction length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("predictions must be finite probabilities")
    return {"rows": len(y), "positives": int(y.sum()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "logloss": float(log_loss(y, p, labels=[0, 1]))}


def evaluate(frame, scores, dataset):
    result = binary_metrics(frame.label, scores)
    frame = frame.assign(score=np.asarray(scores))
    if dataset == "ebnerd":
        groups = []
        for _, group in frame.groupby("group_id", sort=False):
            # Explicit deterministic tie rule; official scorer parity must be
            # verified before comparing these local results with a leaderboard.
            group = group.sort_values(["score", "sample_id"], ascending=[False, True])
            y = group.label.to_numpy()
            if y.sum() == 0 or y.sum() == len(y):
                raise ValueError("EB-NeRD grouped AUC requires both classes in every impression")
            values = {"auc": float(roc_auc_score(y, group.score)),
                      "mrr": float((y / np.arange(1, len(y) + 1)).sum() / y.sum())}
            for k in [5, 10]:
                discount = 1 / np.log2(np.arange(2, min(k, len(y)) + 2))
                values[f"ndcg@{k}"] = float(np.dot(y[:k], discount) /
                                                   np.dot(np.sort(y)[::-1][:k], discount))
            groups.append(values)
        result["impression_macro"] = {key: float(np.mean([g[key] for g in groups])) for key in groups[0]}
        result["groups"] = len(groups)
    elif dataset == "kuairand-1k":
        result["by_policy"] = {str(key): binary_metrics(g.label, g.score)
                               for key, g in frame.groupby("policy")}
        result["by_policy_scene"] = {f"{policy}/{scene}": binary_metrics(g.label, g.score)
                                     for (policy, scene), g in frame.groupby(["policy", "scene"])}
    else:
        raise ValueError("unknown dataset")
    return result
