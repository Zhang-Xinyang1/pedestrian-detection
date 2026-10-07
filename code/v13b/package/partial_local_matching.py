"""Detached, rejectable local correspondence for the V10 auxiliary loss.

Soft transport itself is established prior art (e.g. G2DA); this independent
implementation tests observed foreign-scene targets and identity-margin checks
within our fixed CLIP-CSF training recipe. It does not infer body-part labels.
"""
import torch
from torch.nn import functional as F

MIN_SIMILARITY = 0.2
NEGATIVE_MARGIN = 0.01
MATCH_TEMPERATURE = 0.05
SINKHORN_ITERATIONS = 30


def matching_specification():
    return dict(method='rejectable_foreign_scene_local_correspondence_v1',
        similarity='cosine_normalized_local_features', min_similarity=MIN_SIMILARITY,
        negative_margin=NEGATIVE_MARGIN, temperature=MATCH_TEMPERATURE,
        sinkhorn_iterations=SINKHORN_ITERATIONS, unmatched_option=True,
        negative_candidates='other_batch_identities_prior_observed_local_memory_in_the_same_source_scene',
        unavailable_negative_policy='reject_this_auxiliary_correspondence',
        correspondence_stop_gradient=True, extra_learned_parameters=0,
        loss_normalization='all_eligible_foreign_source_reliability_mass_including_rejected_slots',
        learned_body_part_visibility=False)


@torch.no_grad()
def soft_partial_plan(similarity, allowed):
    """4x4 real matches plus a dustbin row/column; each real slot capacity<=1."""
    batch,k,j=similarity.shape
    if k!=j:
        raise ValueError('Expected equal local-query counts')
    if allowed.shape!=similarity.shape or allowed.dtype!=torch.bool:
        raise ValueError('Invalid matching eligibility')
    scores=similarity.float().new_full((batch,k+1,k+1),MIN_SIMILARITY/MATCH_TEMPERATURE)
    scores[:,:k,:k]=(similarity.float()/MATCH_TEMPERATURE).masked_fill(~allowed,-10000.)
    marginal=scores.new_ones(k+1)
    marginal[-1]=k
    log_marginal=(marginal/(2*k)).log()
    u=scores.new_zeros(batch,k+1);v=u.clone()
    for _ in range(SINKHORN_ITERATIONS):
        u=log_marginal[None]-torch.logsumexp(scores+v[:,None,:],dim=2)
        v=log_marginal[None]-torch.logsumexp(scores+u[:,:,None],dim=1)
    plan=torch.exp(scores+u[:,:,None]+v[:,None,:])[:,:k,:k]*(2*k)
    plan=plan*allowed
    # Finite-iteration projection enforces capacity even before convergence.
    plan=plan/plan.sum(2,keepdim=True).clamp_min(1.)
    plan=plan/plan.sum(1,keepdim=True).clamp_min(1.)
    if not bool(torch.isfinite(plan).all()):
        raise ValueError('Nonfinite local correspondence')
    return plan


@torch.no_grad()
def observed_correspondence(features,labels,scenes,memory,source_weights,foreign):
    """Each source scene is matched separately before any source aggregation."""
    z=F.normalize(features.detach().float(),dim=-1)
    b,k,d=z.shape
    plan=z.new_zeros((b,6,k,k))
    competitor_available=torch.zeros((b,6,k),device=z.device,dtype=torch.bool)
    margin_sum=0.;margin_count=0
    batch_ids=labels.unique(sorted=True)
    for source_scene in range(6):
        eligible=foreign[:,source_scene]
        if not bool(eligible.any()):
            continue
        positive=memory.centers[labels,source_scene].detach()
        similarity=torch.einsum('bkd,bjd->bkj',z,positive)
        observed_ids=batch_ids[memory.mass[batch_ids,source_scene]>=2]
        if not len(observed_ids):
            continue
        negative=memory.centers[observed_ids,source_scene].detach()
        negative_similarity=torch.einsum('bkd,ujd->bkuj',z,negative)
        other=observed_ids[None]!=labels[:,None]
        negative_similarity=negative_similarity.masked_fill(~other[:,None,:,None],-2.)
        hardest=negative_similarity.amax((2,3))
        available=other.any(1)[:,None].expand(-1,k)
        competitor_available[:,source_scene]=available & eligible[:,None]
        margin=similarity-hardest[:,:,None]
        allowed=(similarity>=MIN_SIMILARITY)&(margin>NEGATIVE_MARGIN)
        allowed=allowed & available[:,:,None] & eligible[:,None,None]
        allowed=allowed & (source_weights[:,source_scene,None,:]>0)
        current=soft_partial_plan(similarity,allowed)
        plan[:,source_scene]=current
        if bool(allowed.any()):
            margin_sum+=float((margin*current).sum())
            margin_count+=float(current.sum())
    active=plan.sum((2,3))>0
    source=torch.arange(6,device=z.device)[None]
    cross_view=source//3!=scenes[:,None]//3
    cross_modality=source%3!=scenes[:,None]%3
    denominator=foreign.sum().clamp_min(1)*k
    stats=dict(matched_slot_mass=float(plan.sum()),
        matched_slot_fraction=float(plan.sum()/denominator),
        matched_anchor_fraction=float(active.any(1).float().mean()),
        matched_source_pairs=int(active.sum()),
        matched_cross_view_pairs=int((active&cross_view).sum()),
        matched_cross_modality_pairs=int((active&cross_modality).sum()),
        matched_joint_pairs=int((active&cross_view&cross_modality).sum()),
        competitor_available_fraction=float((competitor_available&foreign[:,:,None]).sum()/denominator),
        mean_matched_identity_margin=margin_sum/max(margin_count,1e-12),
        correspondence_row_capacity_max=float(plan.sum(-1).max()),
        correspondence_column_capacity_max=float(plan.sum(-2).max()))
    return plan.detach(),stats


def matched_local_alignment(features,labels,scenes,memory,source_weights,foreign):
    plan,stats=observed_correspondence(features,labels,scenes,memory,source_weights,foreign)
    live=F.normalize(features.float(),dim=-1)
    source=memory.centers[labels].detach()
    distance=1.-torch.einsum('bkd,bsjd->bskj',live,source)
    numerator=(distance*plan*source_weights[:,:,None,:]).sum((1,2,3))
    denominator=source_weights.sum((1,2)).clamp_min(1e-12)
    per_image=numerator/denominator
    terms=[]
    for scene in scenes.unique(sorted=True):
        eligible=(scenes==scene)&foreign.any(1)
        if bool(eligible.any()):
            terms.append(per_image[eligible].mean())
    loss=torch.stack(terms).mean() if terms else features.float().sum()*0.
    return loss,stats
