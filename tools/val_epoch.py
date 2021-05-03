import numpy as np
from scipy.stats import gmean
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader

logger = logging.get_logger(__name__)

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, cnt):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    
    # for cur_iter, (inputs, bboxs, masks, labels, _, meta) in enumerate(val_loader):
    for cur_iter, output_dict in enumerate(val_loader):
    
        if cfg.EPICKITCHENS.USE_BBOX:
            inputs = output_dict['inputs']
            bboxs = output_dict['bboxs']
            masks = output_dict['masks']
            labels = output_dict['label'] 
            # output_dict['index'] 
            meta = output_dict['metadata'] 
        else:
            inputs = output_dict['inputs']
            labels = output_dict['label'] 
            meta = output_dict['metadata'] 

        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        # for key, val in meta.items():
        #     if isinstance(val, (list,)):
        #         for i in range(len(val)):
        #             val[i] = val[i].cuda(non_blocking=True)
        #     else:
        #         meta[key] = val.cuda(non_blocking=True)
        
        if cur_iter == 0:
            # Metrics save for DETECTION.ENABLE only
            log_ori_boxes = []
            log_metadata = []

            # Metrics for all conditions
            log_preds = []
            log_labels = []
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    log_preds.append([])
                    log_labels.append([])



        if cfg.DETECTION.ENABLE:
            # We won't go in here right?
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"]['narrations'].cpu() # Within a narration stores a list of ids

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            log_preds.append(preds.cpu())
            log_ori_boxes.append(ori_boxes.cpu())
            log_metadata.append(metadata.cpu)
            # val_meter.update_stats(preds.cpu(), ori_boxes.cpu(), metadata.cpu())
        else:
            if cfg.EPICKITCHENS.USE_BBOX:
                bboxs = bboxs.cuda()
                masks = masks.cuda()
                preds = model(inputs, bboxes=bboxs, masks=masks)
            else:
                preds = model(inputs)
            
            # Accumulate prediction and label
            if isinstance(labels, (dict,)):
                log_preds[0].append(preds[0])
                log_preds[1].append(preds[1])
                log_labels[0].append(labels['verb'])
                log_labels[1].append(labels['noun'])

        if cur_iter > 0 and cur_iter % cfg.TRAIN.LOG_ITER == 0:
            if cfg.DETECTION.ENABLE:
                all_preds = []
                all_ori_boxes = []
                all_metadata = []
                # Aggregate into a list
                all_preds = torch.stack(log_preds, dim=0).view(-1)
                all_ori_boxes = torch.stack(log_ori_boxes, dim=0).view(-1)
                all_metadata = torch.stack(log_metadata, dim=0).view(-1)

                val_meter.update_stats(all_preds, all_ori_boxes, all_metadata)
            else:
                if len(log_preds) > 1: # Has 'verb', 'noun' output, assume 2
                    all_preds_verb = torch.stack(log_preds[0], dim=0).view(-1, cfg.MODELS.NUM_CLASSES[0])
                    all_preds.append(all_preds_verb)
                    all_labels_verb = torch.stack(log_preds[0], dim=0).view(-1)
                    all_labels.append(all_labels_verb)

                    all_preds_noun = torch.stack(log_preds[1], dim=0).view(-1, cfg.MODELS.NUM_CLASSES[1])
                    all_preds.append(all_preds_noun)
                    all_labels_noun = torch.stack(log_preds[1], dim=0).view(-1)
                    all_labels.append(all_labels_noun)
                else:
                    all_preds.append(torch.stack(log_preds, dim=0).view(-1))
                    all_labels.append(torch.stack(log_labels, dim=0).view(-1))    

                if len(log_preds) > 1:
                    # Compute the verb accuracies.
                    verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(all_preds[0], all_labels[0], (1, 5))

                    # Combine the errors across the GPUs.
                    if cfg.NUM_GPUS > 1:
                        verb_top1_acc, verb_top5_acc = du.all_reduce([verb_top1_acc, verb_top5_acc])

                    # Copy the errors from GPU to CPU (sync point).
                    verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

                    # Compute the noun accuracies.
                    noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(all_preds[1], all_labels[1], (1, 5))

                    # Combine the errors across the GPUs.
                    if cfg.NUM_GPUS > 1:
                        noun_top1_acc, noun_top5_acc = du.all_reduce([noun_top1_acc, noun_top5_acc])

                    # Copy the errors from GPU to CPU (sync point).
                    noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

                    # Compute the action accuracies.
                    action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((all_preds[0], all_preds[1]),
                                                                                        (all_labels[0], all_labels[1]),
                                                                                        (1, 5))
                    # Combine the errors across the GPUs.
                    if cfg.NUM_GPUS > 1:
                        action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

                    # Copy the errors from GPU to CPU (sync point).
                    action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()

                    val_meter.iter_toc()
                    # Update and log stats.
                    val_meter.update_stats(
                        (verb_top1_acc, noun_top1_acc, action_top1_acc),
                        (verb_top5_acc, noun_top5_acc, action_top5_acc),
                        inputs[0].size(0) * cfg.NUM_GPUS
                    )
                else:
                    # Compute the errors.
                    num_topks_correct = metrics.topks_correct(all_preds, all_labels, (1, 5))

                    # Combine the errors across the GPUs.
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]
                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                    # Copy the errors from GPU to CPU (sync point).
                    top1_err, top5_err = top1_err.item(), top5_err.item()

                    val_meter.iter_toc()
                    # Update and log stats.
                    val_meter.update_stats(
                        top1_err, top5_err, inputs[0].size(0) * cfg.NUM_GPUS
                    )
            val_meter.log_iter_stats(cur_epoch, cur_iter, cnt)
            val_meter.iter_tic()
    # Log epoch stats.
    is_best_epoch = val_meter.log_epoch_stats(cur_epoch, cnt)
    val_meter.reset()
    return is_best_epoch

