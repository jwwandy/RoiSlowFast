from slowfast.datasets.epic_kitchens_bbox.hoa import load_detections, DetectionRenderer
from slowfast.datasets.epickitchens_record import EpicKitchensVideoRecord
import torch
import numpy as np
import os
import pickle

def load_precomputed_bbox(cfg, video_ids):
    bboxs_dict = dict()
    for vid in video_ids:
        path_to_bbox = os.path.join(cfg.EPICKITCHENS.BBOX_ANNOTATIONS_DIR, 'annotations_{}.pkl'.format(vid))
        bboxs = None
        with open(path_to_bbox,'rb') as f:
            bboxs = pickle.load(f)
        
        if cfg.EPICKITCHENS.BBOX_ACTIVE_OBJECT:
            bboxs = 
        
        bboxs_dict[vid] = bboxs


        

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