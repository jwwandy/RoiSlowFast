
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
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, EPICTrainMeter, EPICValMeter

logger = logging.get_logger(__name__)

def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, cnt):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    if cfg.BN.FREEZE:
        model.freeze_fn('bn_statistics')

    train_meter.iter_tic()
    data_size = len(train_loader)

    logger.info("Train loader size: {}".format(data_size))

    #for cur_iter, (inputs, bboxs, masks, labels, _, meta) in enumerate(train_loader):
    for cur_iter, output_dict in enumerate(train_loader):
    
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
        

        # Transfer the data to the current GPU device.
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

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

        else:
            # Perform the forward pass.
            if cfg.EPICKITCHENS.USE_BBOX:
                bboxs = bboxs.cuda()
                masks = masks.cuda()
                preds = model(inputs, bboxes=bboxs, masks=masks)
            else:
                preds = model(inputs)

        if isinstance(labels, (dict,)):
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            # Compute the loss.
            loss_verb = loss_fun(preds[0], labels['verb'])
            loss_noun = loss_fun(preds[1], labels['noun'])
            loss = 0.5 * (loss_verb + loss_noun)
            # check Nan Loss.
            misc.check_nan_losses(loss)
        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            # Compute the loss.
            loss = loss_fun(preds, labels)
            # check Nan Loss.
            misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
        else:
            if isinstance(labels, (dict,)):
                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce(
                        [loss_verb, verb_top1_acc, verb_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_verb, verb_top1_acc, verb_top5_acc = (
                    loss_verb.item(),
                    verb_top1_acc.item(),
                    verb_top5_acc.item(),
                )

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce(
                        [loss_noun, noun_top1_acc, noun_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_noun, noun_top1_acc, noun_top5_acc = (
                    loss_noun.item(),
                    noun_top1_acc.item(),
                    noun_top5_acc.item(),
                )

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]),
                                                                                     (labels['verb'], labels['noun']),
                                                                                     (1, 5))
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, action_top1_acc, action_top5_acc = du.all_reduce(
                        [loss, action_top1_acc, action_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, action_top1_acc, action_top5_acc = (
                    loss.item(),
                    action_top1_acc.item(),
                    action_top5_acc.item(),
                )

                train_meter.iter_toc()
                # Update and log stats.
                train_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    (loss_verb, loss_noun, loss),
                    lr, inputs[0].size(0) * cfg.NUM_GPUS
                )
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

                train_meter.iter_toc()
                # Update and log stats.
                train_meter.update_stats(
                    top1_err, top5_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS
                )
        train_meter.log_iter_stats(cur_epoch, cur_iter, cnt)
        train_meter.iter_tic()
        cnt += 1
    # Log epoch stats.
    print("\n")
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return cnt

