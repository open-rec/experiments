import numpy as np
import pandas as pd
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
        grouped = frame.groupby("group_id", sort=False)
        size = grouped.label.size()
        positives = grouped.label.sum().astype(int)
        negatives = size - positives
        if positives.eq(0).any() or negatives.eq(0).any():
            raise ValueError("EB-NeRD grouped AUC requires both classes in every impression")

        # The rank-sum identity computes tie-aware AUC for every impression in
        # one vectorized pass. It is equivalent to sklearn's binary AUC but
        # avoids hundreds of thousands of small estimator calls.
        ascending_rank = grouped.score.rank(method="average", ascending=True)
        positive_rank_sum = ascending_rank.where(frame.label.eq(1), 0).groupby(
            frame.group_id, sort=False
        ).sum()
        auc = (positive_rank_sum - positives * (positives + 1) / 2) / (
            positives * negatives
        )

        # Ranking metrics retain the documented deterministic order for score
        # ties. A stable whole-frame sort replaces the previous Python loop.
        ranked = frame.sort_values(
            ["group_id", "score", "sample_id"],
            ascending=[True, False, True],
            kind="stable",
        ).copy()
        ranked["rank"] = ranked.groupby("group_id", sort=False).cumcount() + 1
        reciprocal = (ranked.label / ranked["rank"]).groupby(
            ranked.group_id, sort=False
        ).sum()
        mrr = reciprocal / positives.reindex(reciprocal.index)

        metrics = {"auc": float(auc.mean()), "mrr": float(mrr.mean())}
        for k in [5, 10]:
            discount = 1 / np.log2(ranked["rank"] + 1)
            dcg = (ranked.label * discount).where(ranked["rank"] <= k, 0).groupby(
                ranked.group_id, sort=False
            ).sum()
            ideal_discount = np.concatenate(
                ([0.0], np.cumsum(1 / np.log2(np.arange(2, k + 2))))
            )
            ideal = pd.Series(
                ideal_discount[np.minimum(positives, k)], index=positives.index
            )
            metrics[f"ndcg@{k}"] = float((dcg / ideal.reindex(dcg.index)).mean())
        result["impression_macro"] = metrics
        result["groups"] = int(len(size))
    elif dataset == "kuairand-1k":
        result["by_policy"] = {str(key): binary_metrics(g.label, g.score)
                               for key, g in frame.groupby("policy")}
        result["by_policy_scene"] = {f"{policy}/{scene}": binary_metrics(g.label, g.score)
                                     for (policy, scene), g in frame.groupby(["policy", "scene"])}
    else:
        raise ValueError("unknown dataset")
    return result
