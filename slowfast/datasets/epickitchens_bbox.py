from slowfast.datasets.epic_kitchens_bbox.hoa import load_detections, DetectionRenderer
from slowfast.datasets.epickitchens_record import EpicKitchensVideoRecord
import torch
import numpy as np
import os
import pickle


def create_mask_3d(bboxs_len, last_dim):
    max_len = np.max(bboxs_len)
    # true_base = [[True] * last_dim]
    # false_base = [[False] * last_dim]
    # mask = np.array([true_base * leni + false_base * (max_len - leni)  for i,leni in enumerate(bboxs_len)])

    '''
    mask = np.apply_along_axis(lambda x: np.pad(np.ones(x[0]).astype(bool), \
                              (0, max_len-x[0]), 'constant', constant_values=(False, False)), 1, \
                    np.repeat(bboxs_len.reshape((-1,1)),max_len,axis=1))
    '''

    mask = np.array([[True] * leni + [False] * (max_len - leni)  for i,leni in enumerate(bboxs_len)])
    mask = np.repeat(mask.reshape((-1,max_len,1)), last_dim, axis=2)
    return mask


def refine_mask_by_filter_out_zero_mask(bboxs, mask):
    '''
    Filter out bounding-boxes that have no width or no height

    bboxs: (T, N, 4)
    mask: (T, N)
    '''
    if bboxs is None:
        return mask 
    
    # bboxs_4 = bboxs.reshape((-1,4))
    no_width = bboxs[:,:,2] -  bboxs[:,:,0] < 20
    no_height = bboxs[:,:,3] -  bboxs[:,:,1] < 20
    has_no_area = np.logical_or(no_width, no_height)
    has_area = np.logical_not(has_no_area)
    mask = np.logical_and(has_area, mask)

    return has_no_area, mask

def create_mask_2d(bboxs_len):
    max_len = np.max(bboxs_len)
    mask = np.array([[True] * leni + [False] * (max_len - leni)  for i,leni in enumerate(bboxs_len)])
    return mask


def load_precomputed_bbox(cfg, video_ids):
    bboxs_dict = dict()
    mask_dict = dict()
    for vid in video_ids:
        path_to_bbox = os.path.join(cfg.EPICKITCHENS.BBOX_ANNOTATIONS_DIR, 'annotations_{}.pkl'.format(vid))
        bboxs, bboxs_len= None, None #(T, max_len, 7)
        with open(path_to_bbox,'rb') as f:
            bboxs,bboxs_len = pickle.load(f)
        
        T, max_len, last_dim = bboxs.shape  
        
        mask = create_mask_2d(bboxs_len) #np.zeros((T, max_len)).astype(bool)
        if cfg.EPICKITCHENS.BBOX_ACTIVE_OBJECT:
            mask = np.logical_and(bboxs[:,:,1] == 2, bboxs[:,:,2] >= cfg.EPICKITCHENS.BBOX_OBJECT_THRESHOLD)
        else:
            mask_object = np.logical_or(bboxs[:,:,1] == 2, bboxs[:,:,1] == 1)
            mask_object_threshold = np.logical_and(mask_object, bboxs[:,:,2] >= cfg.EPICKITCHENS.BBOX_OBJECT_THRESHOLD)
            mask = np.logical_and(mask, mask_object_threshold)

        if cfg.EPICKITCHENS.BBOX_HAND:
            mask_hand = np.logical_and(bboxs[:,:,1] == 0, bboxs[:,:,2] >= cfg.EPICKITCHENS.BBOX_HAND_THRESHOLD)
            mask = np.logical_or(mask, mask_hand)
        
        
        # mask = np.any(mask, axis=1)
        # bboxs = bboxs[mask]
        
        # mask = np.repeat(mask.reshape((-1,max_len,1)), last_dim, axis=2)
        # bboxs = mask * bboxs
        bboxs_dict[vid] = bboxs[:,:,-4:] #(num_frame_in_video, max_num_frames, 4)
        mask_dict[vid] = mask #(num_frame_in_video, max_num_frames)

    return bboxs_dict, mask_dict
        

def load_all_bbox(visual_data_dir, video_records):
    bbox_loaded = []
    for video_record in video_records:
        path_to_bbox = '{}/{}/hand-objects/{}'.format(visual_data_dir,
                                                    video_record.participant,
                                                    video_record.untrimmed_video_name)
        bboxs = load_detections(path_to_bbox)

        boxes = []

        for idx in range(video_record.start_frame, video_record.end_frame+1):
            frame_bbox = bboxs[idx]
            correspondence_d = frame_bbox.get_hand_object_interactions(
                    object_threshold=0, hand_threshold=0)

            active_object_idx = list(correspondence_d.values())
            
            
            
            for object_idx, obj_detect in enumerate(frame_bbox.objects):
                bbox = obj_detect.bbox 
                score = obj_detect.score
                if object_idx in active_object_idx:
                    boxes.append([bbox.left, bbox.bottom, bbox.right, bbox.top, 2, score, 1])
                else:
                    boxes.append([bbox.left, bbox.bottom, bbox.right, bbox.top, 1, score, 1])
            
            for obj_detect in frame_bbox.hands:
                bbox = obj_detect.bbox 
                score = obj_detect.score
                boxes.append([bbox.left, bbox.bottom, bbox.right, bbox.top, 0, 1, score])

        bbox_loaded.append(np.asarray(boxes).astype(float))
    
    return bbox_loaded


def pack_frame_bbox_raw(cfg, video_record, frame_idx):
    path_to_bbox = '{}/{}/hand-objects/{}.pkl'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
                                                 video_record.participant,
                                                 video_record.untrimmed_video_name)
    bboxs = load_detections(path_to_bbox)

    boxes = []
    acc = 0
    for idx in frame_idx:
        frame_bbox = bboxs[idx]
        correspondence_d = frame_bbox.get_hand_object_interactions(
                object_threshold=0, hand_threshold=0)

        active_object_idx = list(correspondence_d.values())
        
        
        if cfg.EPICKITCHENS.BBOX_OBJECT:
            for object_idx, obj_detect in enumerate(frame_bbox.objects):
                bbox = obj_detect.bbox 
                if obj_detect.score >= cfg.EPICKITCHENS.BBOX_OBJECT_THRESHOLD:
                    boxes.append([acc, bbox.left, bbox.top, bbox.right, bbox.bottom])
                    
        else:
            if cfg.EPICKITCHENS.BBOX_ACTIVE_OBJECT:
                for object_idx in active_object_idx:
                    obj_detect = frame_bbox.objects[object_idx]
                    bbox = obj_detect.bbox 
                    if obj_detect.score >= cfg.EPICKITCHENS.BBOX_OBJECT_THRESHOLD:
                        boxes.append([acc, bbox.left, bbox.top, bbox.right, bbox.bottom])
                        
        
        if cfg.EPICKITCHENS.BBOX_HAND:
            for obj_detect in frame_bbox.hands:
                bbox = obj_detect.bbox 
                if obj_detect.score >= cfg.EPICKITCHENS.BBOX_HAND_THRESHOLD:
                    boxes.append([acc, bbox.left, bbox.top, bbox.right, bbox.bottom])
        
        acc += 1
    
    return np.asarray(boxes)