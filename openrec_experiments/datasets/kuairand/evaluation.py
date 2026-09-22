"""KuaiRand policy and scene metrics."""
from ...evaluation import binary_metrics


def evaluate(frame, result):
    result["by_policy"] = {str(key): binary_metrics(g.label, g.score)
                           for key, g in frame.groupby("policy")}
    result["by_policy_scene"] = {f"{policy}/{scene}": binary_metrics(g.label, g.score)
                                 for (policy, scene), g in frame.groupby(["policy", "scene"])}
    return result
