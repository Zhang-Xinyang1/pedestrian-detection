"""WHU protocol copied from verified SC-LoRA evaluation; model independent."""
import numpy as np
import torch
import torch.nn.functional as F

def retrieval_metrics(qf, gf, query, gallery):
    """Full gallery ranking; exclude ALL same-camera images (official code)."""
    qf = F.normalize(qf.float(), dim=1); gf = F.normalize(gf.float(), dim=1)
    gp = np.array([r[1] for r in gallery]); gc = np.array([r[2] for r in gallery])
    per_query = []
    for i, (_, pid, camera, modality) in enumerate(query):
        order = torch.argsort(qf[i] @ gf.t(), descending=True, stable=True).numpy()
        order = order[gc[order] != camera]
        hits = gp[order] == pid
        if not hits.any():
            continue
        ranks = np.flatnonzero(hits)
        ap = float(((np.arange(len(ranks)) + 1) / (ranks + 1)).mean())
        per_query.append({'ap': ap, 'r1': float(hits[:1].any()), 'r5': float(hits[:5].any()),
                          'r10': float(hits[:10].any()), 'scene': int(camera >= 5) * 3 + modality})
    if not per_query:
        raise ValueError('No valid cross-camera query positives')
    result = {k: float(np.mean([r[k] for r in per_query])) for k in ('ap', 'r1', 'r5', 'r10')}
    scene_maps = {str(s): float(np.mean([r['ap'] for r in per_query if r['scene'] == s]))
                  for s in range(6) if any(r['scene'] == s for r in per_query)}
    result.update(scenes=scene_maps, macro_map=float(np.mean(list(scene_maps.values()))),
                  worst_map=min(scene_maps.values()), valid_queries=len(per_query),
                  skipped_queries=len(query)-len(per_query), queries=len(query), gallery=len(gallery))
    return result

def diagnostic_matrices(qf, gf, query, gallery):
    result = {}
    for name, count, key in [('modality', 3, lambda r: r[3]),
                             ('scene', 6, lambda r: 3*int(r[2]>=5)+r[3])]:
        matrix = []
        for source in range(count):
            row = []
            qi = [i for i,r in enumerate(query) if key(r)==source]
            for target in range(count):
                gi = [i for i,r in enumerate(gallery) if key(r)==target]
                try:
                    metrics = retrieval_metrics(qf[qi], gf[gi], [query[i] for i in qi], [gallery[i] for i in gi])
                    row.append({k: metrics[k] for k in ('ap','r1','valid_queries','skipped_queries')})
                except ValueError:
                    row.append(None)
            matrix.append(row)
        result[name] = matrix
    return result
