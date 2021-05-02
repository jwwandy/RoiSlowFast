import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY
from .epickitchens_record import EpicKitchensVideoRecord
from .epickitchens_bbox import load_precomputed_bbox, refine_mask_by_filter_out_zero_mask

from . import transform as transform
from . import utils as utils
from .frame_loader import pack_frames_to_video_clip

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Epickitchens(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.mode = mode
        self.target_fps = 60
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TRAIN_LIST)]
        elif self.mode == "val":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
        elif self.mode == "test":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        else:
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                                       for file in [self.cfg.EPICKITCHENS.TRAIN_LIST, self.cfg.EPICKITCHENS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )

        self.video_ids = set()
        self._video_records = []
        self._spatial_temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                tup_record = EpicKitchensVideoRecord(tup)
                self.video_ids.add(tup_record.untrimmed_video_name)
                if not (tup_record.num_frames >= self.cfg.EPICKITCHENS.SEGMENT_MIN_LENGTH or \
                    tup_record.num_frames <= self.cfg.EPICKITCHENS.SEGMENT_MAX_LENGTH):
                    continue
                for idx in range(self._num_clips):
                    self._video_records.append(tup_record)
                    self._spatial_temporal_idx.append(idx)
        
        self.video_ids = list(self.video_ids)
        self.bboxs_dict, self.mask_dict = load_precomputed_bbox(self.cfg, self.video_ids)
        
        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epickitchens dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )

    def get_frames_sample_info(self, index):
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            ) # 0,1,2,3,...,_num_clip-1
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1 #center crop
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        
        return temporal_sample_index, spatial_sample_index, min_scale, max_scale, crop_size
    
    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        
        temporal_sample_index, spatial_sample_index, min_scale, max_scale, crop_size = self.get_frames_sample_info(index)
        frames,frame_idx = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index)
        # frames: (T,256,456,C)

        bboxs = None
        mask = None
        if self.cfg.EPICKITCHENS.USE_BBOX:
            vid = self._video_records[index].untrimmed_video_name
            vid_bbox = self.bboxs_dict.get(vid, None)
            mask_bbox = self.mask_dict.get(vid, None)
            if not vid_bbox is None:
                bboxs = vid_bbox[frame_idx]
                mask = mask_bbox[frame_idx] #(T,max_len)
        
        # Scale bbox from [0,1]  to [0,H] or [0,W]
        T,H,W,_ = frames.shape
        bboxs[:,:,0] *= W
        bboxs[:,:,1] *= H
        bboxs[:,:,2] *= W
        bboxs[:,:,3] *= H

        # Perform color normalization.
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames,bboxs = self.spatial_sampling(
            frames,
            bboxs = bboxs,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        label = self._video_records[index].label
        frames = utils.pack_pathway_output(self.cfg, frames)
        metadata = self._video_records[index].metadata
        has_no_area, mask = refine_mask_by_filter_out_zero_mask(bboxs, mask)
        bboxs[has_no_area] = np.array([0.0, 0.0, crop_size, crop_size])
        bboxs[:,:,0] /= W
        bboxs[:,:,1] /= H
        bboxs[:,:,2] /= W
        bboxs[:,:,3] /= H

        fast_bboxs = bboxs.copy()
        fast_mask = mask.copy()
        
        if self.cfg.MODEL.ARCH in self.cfg.MODEL.SINGLE_PATHWAY_ARCH:
            all_bboxs = [torch.FloatTensor(fast_bboxs)]
            all_masks = [torch.FloatTensor(fast_mask)]
        else:
            slow_bboxs = bboxs.copy()
            select_slow_idx = np.linspace(0, frames[1].shape[1]-1, frames[1].shape[1] // self.cfg.SLOWFAST.ALPHA).astype(int)
            slow_bboxs = slow_bboxs[select_slow_idx]

            slow_mask = mask.copy()
            slow_mask = slow_mask[select_slow_idx]

            all_bboxs = [torch.FloatTensor(slow_bboxs), torch.FloatTensor(fast_bboxs)]
            all_masks = [torch.FloatTensor(slow_mask), torch.FloatTensor(fast_mask)]

        return frames, all_bboxs, all_masks, label, index, metadata


    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            bboxs = None,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        # bboxs_4: (T,max_len,4)
        if not bboxs is None:
            bboxs_4 = bboxs.reshape((-1,4))
        else:
            bboxs_4 = None
        if spatial_idx == -1:
            frames, bboxs_4 = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale, boxes=bboxs_4
            )
            frames, bboxs_4 = transform.random_crop(frames, crop_size, boxes=bboxs_4)
            frames, bboxs_4 = transform.horizontal_flip(0.5, frames, boxes=bboxs_4)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, bboxs_4 = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale, boxes=bboxs_4
            )
            frames, bboxs_4 = transform.uniform_crop(frames, crop_size, spatial_idx, boxes=bboxs_4)
        
        if not bboxs is None:
            bboxs = bboxs_4.reshape((len(bboxs), -1, 4))
        
        return frames, bboxs
