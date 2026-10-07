"""Training-only local supervision from observed foreign-scene prototypes.

Independently implemented for this experiment. Local-query and specific-to-shared
learning are standard ideas inspired by SeCap/ViSA/IDKL; no novelty claim is made
for attention, CE, KL, or reliability weighting alone. See README_CN.md.
"""
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from partial_local_matching import matched_local_alignment, matching_specification


QUERIES = 4
LOCAL_WEIGHT = 0.1
ALIGN_WEIGHT = 0.5  # inside LOCAL_WEIGHT
DIVERSITY_WEIGHT = 0.1  # inside LOCAL_WEIGHT
TRANSFER_WEIGHT = 0.05
TEMPERATURE = 2.0
QUERY_TEMPERATURE = 0.07
CLASSIFICATION_SCALE = 16.0
WARMUP_EPOCHS = 5
RAMP_END_EPOCH = 10
MIN_MASS = 2
MASS_CAP = 64
DISPERSION_SCALE = 4.0
MOMENTUM = 0.2


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hashes(module):
    return {name: tensor_sha(value) for name, value in module.state_dict().items()}


def mapping_sha(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def transfer_coefficient(epoch, disabled=False):
    if disabled or epoch <= WARMUP_EPOCHS:
        return 0.0
    return TRANSFER_WEIGHT * min((epoch - WARMUP_EPOCHS) / (RAMP_END_EPOCH - WARMUP_EPOCHS), 1.0)


def balanced_mean(values, valid, scenes):
    """Equal weight to present anchor scenes; never create absent observations."""
    terms = []
    for scene in scenes.unique(sorted=True):
        mask = (scenes == scene) & valid
        if bool(mask.any()):
            terms.append(values[mask].mean())
    return torch.stack(terms).mean() if terms else values.sum() * 0.0


def knowledge_transfer_loss(student_logits, probability, valid, scenes):
    if not bool(valid.any()):
        return student_logits.float().sum() * 0.0
    per_image = F.kl_div(F.log_softmax(student_logits.float() / TEMPERATURE, dim=1),
                        probability.detach(), reduction='none').sum(1) * TEMPERATURE ** 2
    return balanced_mean(per_image, valid, scenes)


class ObservedLocalMemory(nn.Module):
    def __init__(self, classes=500, dim=768, queries=QUERIES):
        super().__init__()
        self.classes, self.dim, self.queries = classes, dim, queries
        self.register_buffer('centers', torch.zeros(classes, 6, queries, dim))
        self.register_buffer('dispersion', torch.zeros(classes, 6, queries))
        self.register_buffer('mass', torch.zeros(classes, 6, dtype=torch.long))
        self.register_buffer('successful_updates', torch.zeros((), dtype=torch.long))
        self.register_buffer('skipped_updates', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, features, labels, scenes, succeeded):
        if not succeeded:
            self.skipped_updates.add_(1)
            return
        z = F.normalize(features.detach().float(), dim=-1)
        if not bool(torch.isfinite(z).all()):
            raise ValueError('Nonfinite local memory input')
        for key in (labels * 6 + scenes).unique(sorted=True).tolist():
            identity, scene = divmod(key, 6)
            mask = (labels == identity) & (scenes == scene)
            selected = z[mask]
            center = F.normalize(selected.mean(0), dim=-1)
            spread = (1.0 - (selected * center).sum(-1)).clamp(0, 2).mean(0)
            if self.mass[identity, scene] > 0:
                old = self.centers[identity, scene]
                drift = (1.0 - (center * old).sum(-1)).clamp(0, 2)
                spread = 0.5 * (spread + drift)
                center = F.normalize((1 - MOMENTUM) * old + MOMENTUM * center, dim=-1)
                spread = (1 - MOMENTUM) * self.dispersion[identity, scene] + MOMENTUM * spread
            self.centers[identity, scene] = center
            self.dispersion[identity, scene] = spread
            self.mass[identity, scene] += int(mask.sum())
        self.successful_updates.add_(1)

    def foreign_targets(self, labels, scenes):
        centers = self.centers[labels].detach()
        mass = self.mass[labels]
        foreign = (mass >= MIN_MASS) & (torch.arange(6, device=scenes.device)[None] != scenes[:, None])
        support = (mass.float().clamp(max=MASS_CAP) / MASS_CAP).sqrt()
        weights = support[..., None] * torch.exp(-DISPERSION_SCALE * self.dispersion[labels])
        weights = weights * foreign[..., None]
        total = weights.sum(1)
        target = F.normalize((centers * weights[..., None]).sum(1) / total.clamp_min(1e-12)[..., None], dim=-1)
        return centers, weights, target.detach(), total > 0, foreign


class LocalObservedAuxiliary(nn.Module):
    """An external auxiliary: no learned parameters or hooks in inference output."""
    def __init__(self, classes=500, dim=768, queries=QUERIES, transfer_disabled=False):
        super().__init__()
        self.classes, self.dim, self.queries = classes, dim, queries
        self.transfer_disabled = bool(transfer_disabled)
        self.shared_queries = nn.Parameter(torch.randn(queries, dim) * .02)
        self.modality_offsets = nn.Parameter(torch.zeros(3, queries, dim))
        self.view_offsets = nn.Parameter(torch.zeros(2, queries, dim))
        self.classifiers = nn.ModuleList([nn.Linear(dim, classes, bias=False) for _ in range(3)])
        self.memory = ObservedLocalMemory(classes, dim, queries)
        self._capture = None
        self._handle = None
        self.history, self.diagnostics = [], []
        self._epoch_rows = []
        self._epoch_steps = self._epoch_skips = 0

    def specification(self):
        return dict(method='matched_observed_foreign_scene_local_supervision_v1', classes=self.classes,
                    feature_dim=self.dim, queries=self.queries, scene_order=['G-RGB','G-NIR','G-TIR','A-RGB','A-NIR','A-TIR'],
                    local_weight=LOCAL_WEIGHT, alignment_weight_inside_local=ALIGN_WEIGHT,
                    diversity_weight_inside_local=DIVERSITY_WEIGHT, transfer_weight=TRANSFER_WEIGHT,
                    transfer_temperature=TEMPERATURE, query_temperature=QUERY_TEMPERATURE,
                    classification_scale=CLASSIFICATION_SCALE, warmup_epochs=WARMUP_EPOCHS,
                    ramp_end_epoch=RAMP_END_EPOCH, transfer_disabled=self.transfer_disabled,
                    min_mass=MIN_MASS, mass_cap=MASS_CAP, dispersion_scale=DISPERSION_SCALE,
                    memory_momentum=MOMENTUM, local_candidate_classes=self.classes,
                    cross_scene_target='same_identity_observed_scenes_excluding_current_scene',
                    teacher='detached_correct_foreign_scene_local_prototype_predictions',
                    reliability='capped_observation_support_and_temporal_feature_dispersion',
                    reliability_is_ground_truth_visibility=False, shared_specific_disentanglement_claim=False,
                    trainable_parameters=sum(p.numel() for p in self.parameters()),
                    inference='auxiliary_absent; original_raw768_raw512_concat_L2',
                    added_backbone_forward=False, missing_scene_completion=False,
                    local_correspondence=matching_specification())

    def attach(self, encoder):
        if self._handle is not None:
            raise ValueError('Auxiliary is already attached')
        def capture(module, args, output):
            if module.training:
                self._capture = output[1][:, 1:]
            else:
                self._capture = None
        self._handle = encoder.register_forward_hook(capture)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._capture = None

    def pop_tokens(self):
        if self._capture is None:
            raise ValueError('Training local tokens are unavailable')
        tokens, self._capture = self._capture, None
        return tokens

    def local_features(self, tokens, scenes):
        tokens = tokens.float()
        q = self.shared_queries[None] + self.modality_offsets[scenes % 3] + self.view_offsets[scenes // 3]
        logits = torch.einsum('bkd,bnd->bkn', F.normalize(q, dim=-1), F.normalize(tokens, dim=-1)) / QUERY_TEMPERATURE
        attention = logits.softmax(-1)
        return torch.einsum('bkn,bnd->bkd', attention, tokens), attention

    def classify(self, features, modalities):
        logits = features.new_zeros(features.shape[:-1] + (self.classes,))
        scaled = CLASSIFICATION_SCALE * F.normalize(features, dim=-1)
        for modality in range(3):
            selected = modalities == modality
            if bool(selected.any()):
                logits[selected] = self.classifiers[modality](scaled[selected])
        return logits

    def compute(self, tokens, labels, scenes, student_logits, epoch):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            features, attention = self.local_features(tokens, scenes)
            logits = self.classify(features, scenes % 3)
            targets = labels[:, None].expand(-1, self.queries)
            ce_per_image = F.cross_entropy(logits.reshape(-1, self.classes), targets.reshape(-1), reduction='none').reshape(-1, self.queries).mean(1)
            ce = balanced_mean(ce_per_image, torch.ones_like(labels, dtype=torch.bool), scenes)
            centers, weights, target, valid, foreign = self.memory.foreign_targets(labels, scenes)
            alignment, matching_stats = matched_local_alignment(
                features,labels,scenes,self.memory,weights,foreign)
            normalized_attention = F.normalize(attention, dim=-1)
            gram = normalized_attention @ normalized_attention.transpose(1, 2)
            off_diagonal = ~torch.eye(self.queries, device=tokens.device, dtype=torch.bool)
            diversity = gram[:, off_diagonal].mean()
            # All teacher targets come from prior successful updates, never the
            # current image or a fabricated missing aerial/modality observation.
            with torch.no_grad():
                teacher_logits = features.new_zeros((len(labels), 6, self.queries, self.classes))
                for modality in range(3):
                    for scene in (modality, modality + 3):
                        teacher_logits[:, scene] = self.classifiers[modality](CLASSIFICATION_SCALE * centers[:, scene])
                correct = teacher_logits.argmax(-1) == labels[:, None, None]
                teacher_weights = weights * correct
                teacher_total = teacher_weights.sum((1, 2))
                probability = (teacher_logits / TEMPERATURE).softmax(-1)
                probability = (probability * teacher_weights[..., None]).sum((1, 2)) / teacher_total.clamp_min(1e-12)[:, None]
                teacher_valid = teacher_total > 0
            coefficient = transfer_coefficient(epoch, self.transfer_disabled)
            if coefficient > 0 and bool(teacher_valid.any()):
                transfer = knowledge_transfer_loss(student_logits,probability,teacher_valid,scenes)
            else:
                transfer = student_logits.float().sum() * 0.0
            local = LOCAL_WEIGHT * (ce + ALIGN_WEIGHT * alignment + DIVERSITY_WEIGHT * diversity)
            loss = local + coefficient * transfer
            foreign_scene = torch.arange(6, device=scenes.device)[None]
            different_view = foreign_scene // 3 != scenes[:, None] // 3
            different_modality = foreign_scene % 3 != scenes[:, None] % 3
            stats = dict(local_ce=float(ce.detach()), cross_scene_alignment=float(alignment.detach()),
                         attention_overlap=float(diversity.detach()), transfer_loss=float(transfer.detach()),
                         transfer_coefficient=coefficient, weighted_loss=float(loss.detach()),
                         local_correct_fraction=float((logits.argmax(-1) == targets).float().mean()),
                         foreign_anchor_fraction=float(valid.any(1).float().mean()),
                         teacher_anchor_fraction=float(teacher_valid.float().mean()),
                         attention_entropy=float((-(attention * attention.clamp_min(1e-12).log()).sum(-1) / math.log(attention.shape[-1])).mean()),
                         foreign_pairs=int(foreign.sum()), cross_view_pairs=int((foreign & different_view).sum()),
                         cross_modality_pairs=int((foreign & different_modality).sum()),
                         joint_pairs=int((foreign & different_view & different_modality).sum()),
                         batch_samples=len(labels))
            stats.update(matching_stats)
            return loss, features, stats

    def begin_epoch(self):
        self._epoch_rows = []
        self._epoch_steps = self._epoch_skips = 0

    def record_batch(self, features, labels, scenes, succeeded, stats):
        self.memory.update(features, labels, scenes, succeeded)
        self._epoch_steps += int(succeeded)
        self._epoch_skips += int(not succeeded)
        self._epoch_rows.append(stats)

    def end_epoch(self, epoch):
        if not self._epoch_rows:
            raise ValueError('Missing auxiliary epoch accounting')
        rows = self._epoch_rows
        record = dict(epoch=epoch, batches=len(rows), successful_updates=self._epoch_steps,
                      skipped_updates=self._epoch_skips, observations=int(self.memory.mass.sum()),
                      observed_identity_count=int((self.memory.mass.sum(1) > 0).sum()),
                      observed_identity_scene_count=int((self.memory.mass > 0).sum()),
                      transfer_coefficient=transfer_coefficient(epoch, self.transfer_disabled))
        for name in rows[0]:
            record['mean_' + name] = sum(row[name] for row in rows) / len(rows)
        self.history.append(record)
        return record

    def observe_gradient(self, loss, model, epoch, batch):
        parameter = dict(model.named_parameters())['image_encoder.transformer.resblocks.11.mlp.c_proj.weight']
        gradient = torch.autograd.grad(loss, parameter, retain_graph=True, allow_unused=True)[0]
        row = dict(epoch=epoch, batch=batch, parameter='image_encoder.transformer.resblocks.11.mlp.c_proj.weight',
                   auxiliary_gradient_norm=0.0 if gradient is None else float(gradient.detach().float().norm()))
        self.diagnostics.append(row)
        return row

    def summary(self):
        hashes = state_hashes(self)
        return dict(specification=self.specification(), final_state_sha256=mapping_sha(hashes), tensor_hashes=hashes,
                    successful_updates=int(self.memory.successful_updates), skipped_updates=int(self.memory.skipped_updates),
                    observations=int(self.memory.mass.sum()), observed_identity_count=int((self.memory.mass.sum(1) > 0).sum()),
                    observed_identity_scene_count=int((self.memory.mass > 0).sum()), epoch_history=self.history,
                    production_transfer_activated=any(r['transfer_coefficient'] > 0 for r in self.history))


def initialize_auxiliary(model, seed, transfer_disabled=False):
    """Restore original Torch RNG streams after auxiliary initialization."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    before_cpu = torch.get_rng_state().clone()
    before_cuda = [v.clone() for v in torch.cuda.get_rng_state_all()] if devices else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(60000 + seed)
        auxiliary = LocalObservedAuxiliary(model.num_classes, 768, transfer_disabled=transfer_disabled)
        with torch.no_grad():
            for head in auxiliary.classifiers:
                head.weight.copy_(model.classifier.weight.detach().cpu())
    assert torch.equal(before_cpu, torch.get_rng_state()), 'Auxiliary changed CPU RNG'
    assert all(torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all())), 'Auxiliary changed CUDA RNG'
    # The approved builder initially keeps classifier/BN layers on CPU while
    # its visual encoder is already on CUDA. Follow the producer of patch
    # tokens before constructing optimizer groups, not the first classifier.
    return auxiliary.to(next(model.image_encoder.parameters()).device)
