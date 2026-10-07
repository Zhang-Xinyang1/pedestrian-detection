import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import json
import math
from prototype_comparison import PrototypeComparisonMemory, prototype_comparison_loss
from l2sp_diagnostics import sample_training_diagnostics, INTERVAL


def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, csf_memory=None, csf_weight=0.0,
             csf_temperature=0.07, csf_reconstruction_weight=0.1,
             prototype_method='csf', starting_point_regularizer=None, local_auxiliary=None):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    csf_weight = float(csf_weight)
    csf_temperature = float(csf_temperature)
    csf_reconstruction_weight = float(csf_reconstruction_weight)
    if csf_weight not in (0.0, 0.05, 0.1, 0.2):
        raise ValueError('Unsupported CSF loss weight')
    if csf_temperature <= 0.0 or csf_reconstruction_weight < 0.0:
        raise ValueError('Invalid CSF loss configuration')
    if csf_memory is None:
        model_ref = model.module if hasattr(model, 'module') else model
        csf_memory = PrototypeComparisonMemory(
            num_classes=int(getattr(model_ref, 'num_classes', 500)),
            feature_dim=768, momentum=0.2, ridge=1e-4,
        ).to(device)
    csf_batches = 0
    csf_loss_sum = 0.0
    csf_valid_sum = 0.0
    csf_factor_target_sum = 0.0
    csf_identity_target_sum = 0.0
    csf_zero_valid_batches = 0
    csf_successful_steps = 0
    csf_skipped_steps = 0
    csf_epoch_history = []
    candidate_sum = 0.0
    evaluation_history = []
    stopped_early = False
    early_stop_reason = None
    final_epoch = 0
    l2sp_epoch_history=[]
    training_diagnostics=[]
    checkpoint_history=[]
    learning_rate_history=[]

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    # Same 500 identity candidates for every image; only the description changes.
    # Missing training scenes never mask an identity out of the denominator.
    from scene_prompt import build_text_banks, matched_scene_logits
    text_features = build_text_banks(model, num_classes, batch)

    for epoch in range(1, epochs + 1):
        final_epoch = epoch
        epoch_csf_batches = 0
        epoch_csf_loss = 0.0
        epoch_csf_valid = 0.0
        epoch_csf_factor = 0.0
        epoch_csf_identity = 0.0
        epoch_zero_valid = 0
        epoch_candidate_sum = 0.0
        epoch_regularization_sum=0.0
        epoch_regularization_batches=0
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()
        from lr_tail_schedule import validate_optimizer_lrs
        lr_record=validate_optimizer_lrs(optimizer,scheduler,epoch)
        learning_rate_history.append(lr_record)

        model.train()
        if local_auxiliary is not None:
            local_auxiliary.train()
            local_auxiliary.begin_epoch()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            scene = target_view.to(device).long()
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                logits = matched_scene_logits(image_features, text_features, scene)
                base_loss = loss_fn(score, feat, target, target_cam, logits)
                loss = base_loss
                identity_feature = feat[1] if isinstance(feat, (list, tuple)) else feat
                if identity_feature.ndim != 2 or identity_feature.shape[1] != 768:
                    raise ValueError('CSF requires the existing 768-D identity feature branch')
                csf_loss, csf_stats = prototype_comparison_loss(
                    identity_feature, target, scene, csf_memory, prototype_method,
                    temperature=csf_temperature,
                    reconstruction_weight=csf_reconstruction_weight,
                )
                loss = loss + csf_weight * csf_loss

            if local_auxiliary is not None:
                auxiliary_loss, local_features, auxiliary_stats = local_auxiliary.compute(
                    local_auxiliary.pop_tokens(), target, scene, score[0], epoch)
                loss = loss + auxiliary_loss
            reg_loss = starting_point_regularizer(model) if starting_point_regularizer is not None else loss*0.0
            loss = loss + reg_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            epoch_regularization_sum += float(reg_loss.detach().cpu())
            epoch_regularization_batches += 1
            probe_now = n_iter % INTERVAL == 0
            if cfg.SOLVER.STAGE2.MAX_EPOCHS == 2 and n_iter + 1 == len(train_loader_stage2):
                probe_now = True  # short smoke observes drift after AMP startup
            if probe_now:
                with amp.autocast(enabled=True):
                    row=sample_training_diagnostics(cfg,model,score,feat,target,logits,base_loss,
                        csf_loss,csf_weight,reg_loss,epoch,n_iter+1)
                training_diagnostics.append(row)
                print('L2SP_TRAIN_DIAGNOSTIC ' + json.dumps(row,sort_keys=True),flush=True)
                if local_auxiliary is not None:
                    local_row=local_auxiliary.observe_gradient(auxiliary_loss,model,epoch,n_iter+1)
                    print('LOCAL_AUX_GRADIENT '+json.dumps(local_row,sort_keys=True),flush=True)
            scaler.scale(loss).backward()

            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            step_succeeded = float(scaler.get_scale()) >= scale_before
            if csf_memory is not None:
                csf_memory.update(
                    identity_feature.detach(), target, scene,
                    step_succeeded=step_succeeded,
                )
            if local_auxiliary is not None:
                local_auxiliary.record_batch(local_features.detach(),target,scene,step_succeeded,auxiliary_stats)

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            if csf_memory is not None:
                batch_csf_loss = float(csf_loss.detach().float().cpu())
                batch_valid = float(csf_stats['valid_anchor_fraction'].detach().float().cpu())
                batch_factor = float(csf_stats['factor_target_fraction'].detach().float().cpu())
                batch_identity = float(csf_stats['identity_target_fraction'].detach().float().cpu())
                batch_candidates = float(csf_stats['mean_candidate_identities'].cpu())
                candidate_sum += batch_candidates
                epoch_candidate_sum += batch_candidates
                csf_batches += 1
                csf_loss_sum += batch_csf_loss
                csf_valid_sum += batch_valid
                csf_factor_target_sum += batch_factor
                csf_identity_target_sum += batch_identity
                csf_zero_valid_batches += int(batch_valid == 0.0)
                csf_successful_steps += int(step_succeeded)
                csf_skipped_steps += int(not step_succeeded)
                epoch_csf_batches += 1
                epoch_csf_loss += batch_csf_loss
                epoch_csf_valid += batch_valid
                epoch_csf_factor += batch_factor
                epoch_csf_identity += batch_identity
                epoch_zero_valid += int(batch_valid == 0.0)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % checkpoint_period == 0:
            from run_official import sha, fingerprints
            from pathlib import Path
            checkpoint=Path(cfg.OUTPUT_DIR)/f'{cfg.MODEL.NAME}_{epoch}.pth'
            cp_record=dict(epoch=epoch,file=checkpoint.name,sha256=sha(checkpoint),parameters=fingerprints(model))
            checkpoint_history.append(cp_record)
            (Path(cfg.OUTPUT_DIR)/f'checkpoint_audit_stage2_{epoch:03d}.json').write_text(
                json.dumps(cp_record,indent=2,sort_keys=True),encoding='utf-8')
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

        if epoch % eval_period == 0 and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
            evaluation_record = dict(
                epoch=epoch,
                map=float(mAP),
                rank1=float(cmc[0]),
            )
            evaluation_history.append(evaluation_record)
            print(
                'SCENE_PROMPT_EVALUATION_CHECK '
                + json.dumps(evaluation_record, sort_keys=True),
                flush=True,
            )

        if csf_memory is not None:
            epoch_denominator = max(epoch_csf_batches, 1)
            epoch_record = dict(
                epoch=epoch, batches=epoch_csf_batches,
                prototype_method=prototype_method,
                mean_candidate_identities=epoch_candidate_sum / epoch_denominator,
                mean_loss=epoch_csf_loss / epoch_denominator,
                mean_valid_anchor_fraction=epoch_csf_valid / epoch_denominator,
                mean_factor_target_fraction=epoch_csf_factor / epoch_denominator,
                mean_identity_target_fraction=epoch_csf_identity / epoch_denominator,
                zero_valid_batches=epoch_zero_valid,
                factor_valid_identity_fraction=float(csf_memory.factor_valid.float().mean().item()),
                identity_initialized_fraction=float(csf_memory.identity_valid.float().mean().item()),
            )
            csf_epoch_history.append(epoch_record)
            print('SCENE_PROMPT_CSF_EPOCH ' + json.dumps(epoch_record, sort_keys=True), flush=True)

        if starting_point_regularizer is not None:
            from pathlib import Path
            epoch_drift=starting_point_regularizer.drift(model)
            record=dict(epoch=epoch,batches=epoch_regularization_batches,
                mean_weighted_penalty=epoch_regularization_sum/epoch_regularization_batches,
                squared_deviation_sum=epoch_drift['squared_deviation_sum'],
                rms_deviation=epoch_drift['rms_deviation'],
                relative_frobenius_deviation=epoch_drift['relative_frobenius_deviation'],
                weighted_penalty_at_epoch_end=epoch_drift['weighted_penalty'])
            l2sp_epoch_history.append(record)
            print('L2SP_EPOCH ' + json.dumps(record,sort_keys=True),flush=True)
            (Path(cfg.OUTPUT_DIR)/'training_diagnostics.json').write_text(
                json.dumps(training_diagnostics,indent=2,sort_keys=True),encoding='utf-8')
        if local_auxiliary is not None:
            from pathlib import Path
            local_record=local_auxiliary.end_epoch(epoch)
            print('LOCAL_OBSERVED_EPOCH '+json.dumps(local_record,sort_keys=True),flush=True)
            (Path(cfg.OUTPUT_DIR)/'local_auxiliary_training_diagnostics.json').write_text(
                json.dumps(local_auxiliary.diagnostics,indent=2,sort_keys=True),encoding='utf-8')
            (Path(cfg.OUTPUT_DIR)/'local_auxiliary_epoch_history.json').write_text(
                json.dumps(local_auxiliary.history,indent=2,sort_keys=True),encoding='utf-8')
        from pathlib import Path
        (Path(cfg.OUTPUT_DIR)/'learning_rate_epoch_history.json').write_text(
            json.dumps(learning_rate_history,indent=2,sort_keys=True),encoding='utf-8')
        if epoch % checkpoint_period == 0:
            from training_state_io import save_training_state
            save_training_state(Path(cfg.OUTPUT_DIR)/'training_state_latest.pth',
                model,optimizer,optimizer_center,scheduler,scaler,csf_memory,local_auxiliary,
                epoch,dict(actual_batches_per_epoch=list(train_loader_stage2.counts),
                successful_updates=csf_successful_steps,skipped_updates=csf_skipped_steps,
                learning_rate_history=learning_rate_history,prototype_history=csf_epoch_history,
                l2sp_epoch_history=l2sp_epoch_history,evaluation_history=evaluation_history),cfg)
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)
    denominator = max(csf_batches, 1)
    if csf_memory is not None:
        memory_diagnostics = csf_memory.batch_diagnostics(
            torch.empty(0, dtype=torch.long, device=device)
        )
        if not math.isfinite(memory_diagnostics['mean_condition']):
            memory_diagnostics['mean_condition'] = None
        factor_metrics = csf_memory.factor_metrics()
        memory_state_sha256 = csf_memory.state_sha256()
    else:
        memory_diagnostics = {}
        factor_metrics = {}
        memory_state_sha256 = None
    summary = dict(
        weight=csf_weight, temperature=csf_temperature,
        prototype_method=prototype_method,
        candidate_policy=('full_observed_identity_center' if prototype_method == 'identity_full' else
                          ('full_identity_with_factor_fallback' if prototype_method == 'csf_full_identity'
                          else 'matched_observed_scene_full_rank')),
        mean_candidate_identities=candidate_sum / denominator,
        reconstruction_weight=csf_reconstruction_weight, batches=csf_batches,
        mean_loss=csf_loss_sum / denominator,
        mean_valid_anchor_fraction=csf_valid_sum / denominator,
        mean_factor_target_fraction=csf_factor_target_sum / denominator,
        mean_identity_target_fraction=csf_identity_target_sum / denominator,
        zero_valid_batches=csf_zero_valid_batches,
        successful_optimizer_steps=csf_successful_steps,
        skipped_optimizer_steps=csf_skipped_steps,
        feature_branch='identity_768', feature_dim=768,
        trainable_parameters=0, data_and_sampler_unchanged=True,
        epoch_history=csf_epoch_history, memory=memory_diagnostics,
        factor_metrics=factor_metrics, memory_state_sha256=memory_state_sha256,
        stopped_early=stopped_early, final_epoch=final_epoch,
        early_stop_reason=early_stop_reason,
        l2sp_epoch_history=l2sp_epoch_history,
        checkpoint_history=checkpoint_history,
        evaluation_history=evaluation_history,
        learning_rate_history=learning_rate_history,
        early_stop_policy=dict(
            enabled=False, evaluation_period=10,
            rule='disabled: complete configured epoch budget'))
    print('SCENE_PROMPT_CSF_DONE ' + json.dumps(summary, sort_keys=True), flush=True)
    return summary, csf_memory


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
