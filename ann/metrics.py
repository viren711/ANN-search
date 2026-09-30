"""k-recall@k as defined in the DiskANN paper: |A_k(q) ∩ G_k(q)| / k averaged over queries."""
import numpy as np


def k_recall_at_k(approx_ids, gt_ids, k):
    a = np.asarray(approx_ids)[:, :k]
    g = np.asarray(gt_ids)[:, :k]
    assert a.shape[0] == g.shape[0], "query count mismatch"
    hits = (a[:, :, None] == g[:, None, :]).any(axis=2).sum(axis=1)
    return float(hits.mean() / k)


def recall_report(approx_ids, gt_ids):
    return {"1-recall@1": k_recall_at_k(approx_ids, gt_ids, 1),
            "10-recall@10": k_recall_at_k(approx_ids, gt_ids, 10)}
