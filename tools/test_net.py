#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import os
import pickle
import numpy as np
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter, EPICTestMeter
logger = logging.get_logger(__name__)


def to_cuda(data):
    if isinstance(data, (list,)):
        for i in range(len(data)):
            data[i] = data[i].cuda(non_blocking=True)
    else:
        data = data.cuda(non_blocking=True)
    return data

def perform_test(test_loader, model, test_meter, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    if not cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
        x_feat_list = [[],[]]
    elif cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
        x_feat_list = []
    # for cur_iter, (inputs, bboxs, masks, labels, video_idx, meta) in enumerate(test_loader):
    for cur_iter, output_dict in enumerate(test_loader):
        if cur_iter % 100 == 0:
            logger.info("Testing iter={}".format(cur_iter))
    
        inputs = output_dict['inputs']
        labels = output_dict['label'] 
        video_idx = output_dict['index'] 
        meta = output_dict['metadata'] 
        if cfg.EPICKITCHENS.USE_BBOX:
            bboxs = output_dict['bboxs']
            masks = output_dict['masks']
        else:
            bboxs = None
            masks = None

        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        # Transfer the data to the current GPU device.
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        video_idx = video_idx.cuda()

        if cfg.DETECTION.ENABLE:
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach().cpu(),
                ori_boxes.detach().cpu(),
                metadata.detach().cpu(),
            )
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            if cfg.EPICKITCHENS.USE_BBOX:
                bboxs = to_cuda(bboxs)
                masks = to_cuda(masks)
                preds_pair = model(inputs, bboxes=bboxs, masks=masks)
            else:
                preds_pair = model(inputs)
            
            if cfg.TEST.EXTRACT_FEATURES:
                preds, x_feat = preds_pair
            else:
                preds = preds_pair

            if isinstance(labels, (dict,)):
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    verb_preds, verb_labels, video_idx = du.all_gather(
                        [preds[0], labels['verb'], video_idx]
                    )

                    noun_preds, noun_labels, video_idx = du.all_gather(
                        [preds[1], labels['noun'], video_idx]
                    )
                    meta = du.all_gather_unaligned(meta)
                    metadata = {'narration_id': []}
                    for i in range(len(meta)):
                        metadata['narration_id'].extend(meta[i]['narration_id'])
                    
                    
                    if not cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
                        x_feat_slow, x_feat_fast = du.all_gather([x_feat[0], x_feat[1]])
                        #print(x_feat_slow.shape, x_feat_fast.shape)
                        ##torch.Size([8, 2048, 8, 7, 7]) torch.Size([8, 256, 32, 7, 7])
                        x_feat_list[0] += [x_feat_slow]
                        x_feat_list[1] += [x_feat_fast]
                    elif cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
                        x_feat = du.all_gather([x_feat])
                        x_feat_list.append(x_feat[0])
                else:
                    metadata = meta
                    verb_preds, verb_labels, video_idx = preds[0], labels['verb'], video_idx
                    noun_preds, noun_labels, video_idx = preds[1], labels['noun'], video_idx
                    if not cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
                        x_feat_list[0].append(x_feat[0])
                        x_feat_list[1].append(x_feat[1])
                    elif cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
                        x_feat_list.append(x_feat)
                    
                    
                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    (verb_preds.detach().cpu(), noun_preds.detach().cpu()),
                    (verb_labels.detach().cpu(), noun_labels.detach().cpu()),
                    metadata,
                    video_idx.detach().cpu(),
                )
                # test_meter.log_iter_stats(cur_iter)
            else:
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    preds, labels, idx = du.all_gather(
                        [preds, labels, video_idx]
                    )

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach().cpu(),
                    labels.detach().cpu(),
                    video_idx.detach().cpu(),
                )
                # test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if cfg.TEST.DATASET == 'epickitchens':
        preds, labels, metadata = test_meter.finalize_metrics()
    else:
        test_meter.finalize_metrics()
        preds, labels, metadata = None, None, None
    test_meter.reset()
    
    if cfg.TEST.EXTRACT_FEATURES:
        if not cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
            final_feat_list = [[],[]]
            final_feat_list[0] = [t.cpu() for t in x_feat_list[0]]
            final_feat_list[1] = [t.cpu() for t in x_feat_list[1]]
        elif cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
            final_feat_list = [t.cpu() for t in x_feat_list]
        
        return preds, labels, metadata, final_feat_list
    else:
        return preds, labels, metadata

def test_from_train(model, cfg, cnt=-1):
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        if cfg.TEST.DATASET == 'epickitchens':
            test_meter = EPICTestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
            )
        else:
            test_meter = TestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
            )

    # # Perform multi-view test on the entire dataset.
    scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
    if not os.path.exists(scores_path):
        os.makedirs(scores_path)
    
    filename_root = cfg.EPICKITCHENS.TEST_LIST.split('.')[0]
    if cnt >= 0:
        file_name = '{}_{}_{}.pkl'.format(filename_root, cnt, cfg.MODEL.MODEL_NAME)
    else:
        file_name = '{}_{}_{}.pkl'.format(filename_root, 'test_only', cfg.MODEL.MODEL_NAME)
    file_path = os.path.join(scores_path, file_name)
    logger.info(file_path)

    pickle.dump([], open(file_path, 'wb+'))
    preds, labels, metadata = perform_test(test_loader, model, test_meter, cfg)

    if du.is_master_proc():
        if cfg.TEST.DATASET == 'epickitchens':
            results = {'verb_output': preds[0],
                       'noun_output': preds[1],
                       'verb_gt': labels[0],
                       'noun_gt': labels[1],
                       'narration_id': metadata}
            scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            pickle.dump(results, open(file_path, 'wb'))

def test(cfg, cnt=-1):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # # Perform multi-view test on the entire dataset.
    scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
    if not os.path.exists(scores_path):
        os.makedirs(scores_path)
    
    filename_root = cfg.EPICKITCHENS.TEST_LIST.split('.')[0]
    if cnt >= 0:
        file_name = '{}_{}_{}.pkl'.format(filename_root, cnt, cfg.MODEL.MODEL_NAME)
    else:
        file_name = '{}_{}_{}.pkl'.format(filename_root, 'test_only', cfg.MODEL.MODEL_NAME)
    file_path = os.path.join(scores_path, file_name)
    logger.info(file_path)

    # Print config.
    # if cnt < 0:
    #     logger.info("Test with config:")
    #     logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.EPICKITCHENS.USE_BBOX:
        model.load_weight_slowfast()

    # if du.is_master_proc():
    #     misc.log_model_info(model, cfg, is_train=False)
    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    # Create video testing loaders.
    if cfg.TEST.EXTRACT_FEATURES_MODE != "" and cfg.TEST.EXTRACT_FEATURES_MODE in ["test","train","val"]:
        test_loader = loader.construct_loader(cfg, cfg.TEST.EXTRACT_FEATURES_MODE)
    else:
        test_loader = loader.construct_loader(cfg, "test")

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            len(test_loader.dataset)
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        if cfg.TEST.DATASET == 'epickitchens':
            test_meter = EPICTestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
            )
        else:
            test_meter = TestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
            )

    
    pickle.dump([], open(file_path, 'wb+'))
    if cfg.TEST.EXTRACT_FEATURES:
        preds, labels, metadata, x_feat_list = perform_test(test_loader, model, test_meter, cfg)
    else:
        preds, labels, metadata = perform_test(test_loader, model, test_meter, cfg)

    if du.is_master_proc():
        if cfg.TEST.DATASET == 'epickitchens':
            results = {'verb_output': preds[0],
                       'noun_output': preds[1],
                       'verb_gt': labels[0],
                       'noun_gt': labels[1],
                       'narration_id': metadata}
            scores_path = os.path.join(cfg.OUTPUT_DIR, 'scores')
            if not os.path.exists(scores_path):
                os.makedirs(scores_path)
            pickle.dump(results, open(file_path, 'wb'))

            if cfg.TEST.EXTRACT_FEATURES:
                pid = cfg.EPICKITCHENS.FEATURE_VID.split("_")[0]
                if not os.path.exists(os.path.join(cfg.TEST.EXTRACT_FEATURES_PATH, pid)):
                    os.mkdir(os.path.join(cfg.TEST.EXTRACT_FEATURES_PATH, pid))
                if not cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
                    arr_slow = torch.cat(x_feat_list[0], dim=0).numpy()
                    arr_fast = torch.cat(x_feat_list[1], dim=0).numpy()
                    print(arr_slow.shape, arr_fast.shape)
                    fpath_feat = os.path.join(cfg.TEST.EXTRACT_FEATURES_PATH, pid, '{}.pkl'.format(cfg.EPICKITCHENS.FEATURE_VID))
                    with open(fpath_feat,'wb+') as f:
                        pickle.dump([arr_slow, arr_fast], f)
                elif cfg.TEST.EXTRACT_MSTCN_FEATURES and cfg.TEST.EXTRACT_FEATURES:
                    fpath_feat = os.path.join(cfg.TEST.EXTRACT_FEATURES_PATH, pid, '{}.npy'.format(cfg.EPICKITCHENS.FEATURE_VID))
                    with open(fpath_feat,'wb+') as f:
                        arr = torch.cat(x_feat_list, dim=0).numpy()
                        print(arr.shape)
                        np.save(f, arr)
                    