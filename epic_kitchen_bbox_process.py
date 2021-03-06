import pandas as pd 
import numpy as np
import os 
import argparse
import sys
import torch
import pickle
from slowfast.datasets.epickitchens_record import EpicKitchensVideoRecord
from slowfast.datasets.epic_kitchens_bbox.hoa import load_detections, DetectionRenderer
from tqdm import tqdm
from slowfast.datasets.epickitchens_bbox import load_all_bbox

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--annotations_dir', type=str)
parser.add_argument('--visual_data_dir', type=str)
parser.add_argument('--bbox_annotations_dir', type=str)
parser.add_argument('--anno_format', type=str)
parser.add_argument('--anno_file', type=str)
# parser.add_argument('--pid', type=str)

args = parser.parse_args()

'''
python epic_kitchen_bbox_process.py --annotations_dir '/raid/xiaoyuz1/EPIC/epic-annotations' --visual_data_dir '/raid/xiaoyuz1/EPIC/EPIC-KITCHENS' --bbox_annotations_dir '/raid/xiaoyuz1/EPIC/epic-bbox-annotations/all' --anno_file /raid/xiaoyuz1/EPIC/epic-annotations/annotations_slowfast_val2.pkl --anno_format 'annotations_{}.pkl'

python epic_kitchen_bbox_process.py --annotations_dir '/raid/xiaoyuz1/EPIC/epic-annotations' --visual_data_dir '/raid/xiaoyuz1/EPIC/EPIC-KITCHENS' --bbox_annotations_dir '/raid/xiaoyuz1/EPIC/epic-bbox-annotations/all' --anno_file /raid/xiaoyuz1/EPIC/epic-annotations/annotations_slowfast_train2.pkl --anno_format 'annotations_{}.pkl'

'''

# os.path.join(args.annotations_dir, 'EPIC_100_train.pkl')
# pkl_train = pd.read_pickle('/home/xiaoyuz1/epic-kitchens-100-annotations/EPIC_100_train.pkl')
# pkl_val = pd.read_pickle('/home/xiaoyuz1/epic-kitchens-100-annotations/EPIC_100_validation.pkl')


# 'annotations_train_{}.pkl'
def process(fname):
    pkl = pd.read_pickle(fname)   
    vids = pkl.video_id.unique()

    # for video_idx in tqdm(range(len(video_records))):
    for vid_idx in tqdm(range(len(vids))):
        vid = vids[vid_idx]
        pid = vid.split('_')[0]
        path_to_bbox = '{}/{}/hand-objects/{}.pkl'.format(args.visual_data_dir,
                                                 pid,
                                                 vid)
        bboxs = load_detections(path_to_bbox)

        boxes = []
        boxes_len = []
        frame_idx_acc = 0

        for idx in range(len(bboxs)):
            frame_bbox_object = bboxs[idx]
            frame_idx = frame_bbox_object.frame_number
            assert frame_idx_acc <= frame_idx
            
            frame_bbox = []
            
            for _ in range(frame_idx_acc, frame_idx):
                boxes.append([])
                boxes_len.append(0)
                frame_idx_acc += 1

            correspondence_d = frame_bbox_object.get_hand_object_interactions(
                    object_threshold=0, hand_threshold=0)

            if len(correspondence_d) == 0:
                active_object_idx = []
            else:
                active_object_idx = list(correspondence_d.values())
            
            for object_idx, obj_detect in enumerate(frame_bbox_object.objects):
                bbox = obj_detect.bbox 
                if object_idx in active_object_idx:
                    frame_bbox.append([frame_idx_acc, 2.0, obj_detect.score, bbox.left, bbox.top, bbox.right, bbox.bottom])
                else:
                    frame_bbox.append([frame_idx_acc, 1.0, obj_detect.score, bbox.left, bbox.top, bbox.right, bbox.bottom])

                        
            for obj_detect in frame_bbox_object.hands:
                bbox = obj_detect.bbox 
                frame_bbox.append([frame_idx_acc, 0.0, obj_detect.score, bbox.left, bbox.top, bbox.right, bbox.bottom])
               
            if len(frame_bbox) == 0:
                frame_bbox.append([frame_idx, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
            
            boxes.append(frame_bbox)
            boxes_len.append(len(frame_bbox))
            frame_idx_acc += 1
        
        max_num_bbox_per_frame = np.max(boxes_len)
        padded_boxes = []
        for frame_idx, frame_bbox in enumerate(boxes):
            frame_bbox_padded = frame_bbox + [[frame_idx, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]] * (max_num_bbox_per_frame - boxes_len[frame_idx])
            padded_boxes.append(frame_bbox_padded)
    
        with open(os.path.join(args.bbox_annotations_dir, args.anno_format.format(vid)), 'wb+') as f:
            pickle.dump((np.asarray(padded_boxes), np.asarray(boxes_len)), f)
        

# for pid in pkl_train.participant_id.unique():
#     process('annotations_train_{}.pkl', pid)

# for pid in pkl_val.participant_id.unique():
#     process('annotations_val_{}.pkl', pid)
process(args.anno_file)