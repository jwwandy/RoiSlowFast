
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
# from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, EPICTrainMeter, EPICValMeter
import wandb

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

    # train_meter.iter_tic()
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
        
        # Reset the accumulator for calculating accuracy at iter 1,LOG_PERIOD+1...
        # if cur_iter == 0 or cur_iter % cfg.LOG_PERIOD == 1:
        #     if cur_iter == 1:
        #         continue
        if cur_iter % (cfg.LOG_PERIOD * cfg.ACC_LOG_PERIOD_RATIO) == 0:
            log_preds = []
            log_labels = []
            log_loss = []            
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    log_preds.append([])
                    log_labels.append([])
                for i in range(len(inputs)+1):
                    log_loss.append([]) 

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
            
            # if len(preds) > 1:
            #     log_preds[0].append(preds[0])
            #     log_preds[1].append(preds[1])
            # else:
            #     log_preds.append(preds)
                

        if isinstance(labels, (dict,)):
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            # Compute the loss.
            loss_verb = loss_fun(preds[0], labels['verb'])
            loss_noun = loss_fun(preds[1], labels['noun'])
            loss = 0.5 * (loss_verb + loss_noun)
            # check Nan Loss.
            misc.check_nan_losses(loss)
            
            log_loss[0].append(loss_verb.item())
            log_loss[1].append(loss_noun.item())
            log_loss[2].append(loss.item())

            # log_labels[0].append(labels['verb'])
            # log_labels[1].append(labels['noun'])
        else:
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
            # Compute the loss.
            loss = loss_fun(preds, labels)
            # check Nan Loss.
            misc.check_nan_losses(loss.item())

            log_loss.append(loss)
            # log_labels.append(labels)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()
        
        # Add prediction and labels into list
        # Only log loss for DETECTION.ENABLE conditions
        if not cfg.DETECTION.ENABLE:
            if isinstance(labels, (dict,)):
                # Store prediciton and labels in an epoch
                if cfg.NUM_GPUS > 1:
                    preds_v, preds_n, label_v, label_n = du.all_reduce(
                        [preds[0], preds[1], labels['verb'], labels['noun']]
                    )
                else:
                    preds_v, preds_n, label_v, label_n = preds[0], preds[1], labels['verb'], labels['noun']
                log_preds[0].append(preds_v)
                log_preds[1].append(preds_n)
                log_labels[0].append(label_v)
                log_labels[1].append(label_n)
            else:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_reduce(
                        [preds, labels]
                    )

                log_preds.append(preds)
                log_labels.append(labels)
        

        # Aggegrate LOG_ITR * BATCH_SIZE number of samples then compute metrics
        if (cur_iter+1) % cfg.LOG_PERIOD == 0:
            if cfg.DETECTION.ENABLE:
                mean_loss = np.mean(log_loss)
                train_meter.update_stats(None, None, None, mean_loss, lr)
            else:
                all_preds = []
                all_labels = []
                # Aggregate into a list
                if len(log_preds) > 1: # Has 'verb', 'noun' output, assume 2
                    all_preds.append(torch.stack(log_preds[0], dim=0).view(-1, cfg.MODEL.NUM_CLASSES[0]))
                    all_labels.append(torch.stack(log_labels[0], dim=0).view(-1))

                    all_preds.append(torch.stack(log_preds[1], dim=0).view(-1, cfg.MODEL.NUM_CLASSES[1]))
                    all_labels.append(torch.stack(log_labels[1], dim=0).view(-1))

                else:
                    all_preds.append(torch.stack(log_preds, dim=0).view(-1, cfg.MODEL.NUM_CLASSES[0]))
                    all_labels.append(torch.stack(log_labels, dim=0).view(-1))    
                
                if len(all_preds) > 1:
                    loss_verb = np.mean(log_loss[0])
                    loss_noun = np.mean(log_loss[1])
                    loss = np.mean(log_loss[2])
                    # Compute the verb accuracies.
                    # verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))
                    verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(all_preds[0], all_labels[0], (1, 5))

                    predicted_answer_softmax = torch.nn.Softmax(dim=1)(preds[0])
                    predicted_answer_max = torch.max(predicted_answer_softmax.data, 1).indices
                    # print(cnt, predicted_answer_max, labels['verb'])

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
                    # log_loss.append(loss.item())

                    # Compute the noun accuracies.
                    noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(all_preds[1], all_labels[1], (1, 5))

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
                    action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((all_preds[0], all_preds[1]),
                                                                                        (all_labels[0], all_labels[1]),
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

                    wandb_dict = {
                        "train/verb_top1_acc": verb_top1_acc,
                        "train/verb_top5_acc": verb_top5_acc,
                        "train/noun_top1_acc": noun_top1_acc,
                        "train/noun_top5_acc": noun_top5_acc,
                        "train/top1_acc": action_top1_acc,
                        "train/top5_acc": action_top5_acc,
                        "train/verb_loss": loss_verb,
                        "train/noun_loss": loss_noun,
                        "train/loss": loss,
                    }
                    logger.info("\n======>Epoch-{}-Iter-{}-Cnt-{} train/verb_top1_acc: {} \n\ttrain/verb_top5_acc: {}\n\ttrain/top1_acc: {}\n\ttrain/top5_acc: {}\n\ttrain/verb_loss: {}\n\ttrain/noun_loss: {}\n\ttrain/loss: {}".format(cur_epoch, cur_iter,cnt, verb_top1_acc, verb_top5_acc, action_top1_acc, action_top5_acc, loss_verb, loss_noun, loss))
                    wandb.log(wandb_dict, step=cnt)

                    # train_meter.iter_toc()
                    # # Update and log stats.
                    # train_meter.update_stats(
                    #     (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    #     (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    #     (loss_verb, loss_noun, loss),
                    #     lr, inputs[0].size(0) * cfg.NUM_GPUS
                    # )
                else:
                    # Compute the errors.
                    loss = np.mean(log_loss)
                    num_topks_correct = metrics.topks_correct(all_preds, all_labels, (1, 5))
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

                    # train_meter.iter_toc()
                    # # Update and log stats.
                    # train_meter.update_stats(
                    #     top1_err, top5_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS
                    # )
        # train_meter.log_iter_stats(cur_epoch, cur_iter, cnt)
        cnt += 1
        if (cur_iter+1) % cfg.LOG_PERIOD == 0:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # train_meter.iter_tic()
    # Log epoch stats.
    # train_meter.log_epoch_stats(cur_epoch)
    # train_meter.reset()
    return cnt

