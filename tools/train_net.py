#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

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
from train_epoch import train_epoch
from val_epoch import eval_epoch
from test_net import test_from_train

from slowfast.models.video_model_builder import SlowFast

logger = logging.get_logger(__name__)
# torch.autograd.set_detect_anomaly(True) 
def to_cuda(data):
    if isinstance(data, (list,)):
        for i in range(len(data)):
            data[i] = data[i].cuda(non_blocking=True)
    else:
        data = data.cuda(non_blocking=True)
    return data


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, bboxs, masks, labels, _, meta in loader:
            #for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.EPICKITCHENS.USE_BBOX and not cu.has_checkpoint(cfg.OUTPUT_DIR):
        slow_fast_model = SlowFast(cfg)
        if cfg.EPICKITCHENS.USE_BBOX and cfg.EPICKITCHENS.LOAD_SLOWFAST_PRETRAIN:
            if cfg.EPICKITCHENS.SLOWFAST_PRETRAIN_CHECKPOINT_FILE_PATH != "":
                _ = cu.load_checkpoint(
                    cfg.EPICKITCHENS.SLOWFAST_PRETRAIN_CHECKPOINT_FILE_PATH,
                    slow_fast_model,
                    cfg.NUM_GPUS > 1,
                    optimizer=None,
                    inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
                    convert_from_caffe2=False,
                )
        # cfg.TRAIN.CHECKPOINT_TYPE == "caffe2"
        model.load_weight_slowfast(slow_fast_model)
        # model.load_weight_slowfast()

    # if du.is_master_proc():
    #     misc.log_model_info(model, cfg, is_train=True)

    if cfg.BN.FREEZE:
        model.freeze_fn('bn_parameters')
    if cfg.EPICKITCHENS.USE_BBOX:
        model.freeze_fn('slowfast_bbox')

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "" and not cfg.TRAIN.FINETUNE:
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "" and cfg.TRAIN.FINETUNE:
        logger.info("Load from given checkpoint file. Finetuning.")
        _ = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = 0
    else:
        start_epoch = 0
    

    # Create the video train and val loaders.
    if cfg.TRAIN.DATASET != 'epickitchens' or not cfg.EPICKITCHENS.TRAIN_PLUS_VAL:
        train_loader = loader.construct_loader(cfg, "train")
        val_loader = loader.construct_loader(cfg, "val")
    else:
        train_loader = loader.construct_loader(cfg, "train+val")
        val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        if cfg.TRAIN.DATASET == 'epickitchens':
            train_meter = EPICTrainMeter(len(train_loader), cfg)
            val_meter = EPICValMeter(len(val_loader), cfg)
        else:
            train_meter = TrainMeter(len(train_loader), cfg)
            val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    cnt = 0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        cnt = train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, cnt)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            is_best_epoch = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, cnt)
            if is_best_epoch:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, is_best_epoch=is_best_epoch)
            test_from_train(model, cfg, cnt=cnt)

