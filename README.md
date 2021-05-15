# EPIC-KITCHENS-100 SlowFast Networks with ROI

We have taken code from:
- Original SlowFast [repo](https://github.com/facebookresearch/SlowFast) 
```
@ARTICLE{Damen2020RESCALING,
   title={Rescaling Egocentric Vision},
   author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and and Furnari, Antonino 
           and Ma, Jian and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan 
           and Perrett, Toby and Price, Will and Wray, Michael},
           journal   = {CoRR},
           volume    = {abs/2006.13256},
           year      = {2020},
           ee        = {http://arxiv.org/abs/2006.13256},
} 
```
and
```
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```
- A SlowFast repo trained on EPIC-KITCHENS [repo](https://github.com/epic-kitchens/epic-kitchens-slowfast/tree/master/slowfast/models)

## Download EPIC-KITCHENS
- From [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts)

`python epic_downloader.py --rgb-frames --masks`

WARNING: download each participant separately, expect 1-2 days to download the entire dataset. 
- Original annotations from [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations)

## Download other necessary files
- Pretrained model `gdown --id 1cF9MlU7YhGTXn5KZVTklSYc1oypjUeaK`
- Annotations separate by participants and video ids (processed ourselves) `gdown --id 1yeMFdejhz-1l439SSZBwNz9n0iAc5sLP`
- Processed bounding box annotations `gdown --id `

## Example scripts

### Finetuning:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=[PATH TO YOUR RoiSlowFast CODE] python tools/run_net.py --cfg [PATH TO YOUR RoiSlowFast CODE]/epic_config/ALPHA_4_BETA_8_ALL_BBOX_LOCAL_DATASET.yaml --init_method tcp://localhost:[LOCALHOST-NUMBER OF YOUR CHOICE] NUM_GPUS [NUMBER OF GPUS AVAILABLE] OUTPUT_DIR [OUTPUT DIRECTORY PATH TO SAVE CHECKPOINTS] ENABLE_WANDB False EPICKITCHENS.ROI_BRANCH 3 EPICKITCHENS.VISUAL_DATA_DIR [EPIC-KITCHENS BASE DIRECTORY/EPIC-KITCHENS] EPICKITCHENS.ANNOTATIONS_DIR [DIRECTORY THAT CONTAINS DOWNLOADED ANNOTATIONS/epic-annotations] EPICKITCHENS.BBOX_ANNOTATIONS_DIR [DIRECTORY THAT CONTAINS DOWNLOADED BOUNDING BOX ANNOTATIONS/epic-annotations] EPICKITCHENS.TEST_LIST annotations_slowfast_val2.pkl EPICKITCHENS.TRAIN_LIST annotations_slowfast_train2.pkl EPICKITCHENS.VAL_LIST annotations_slowfast_val2.pkl EPICKITCHENS.SLOWFAST_PRETRAIN_CHECKPOINT_FILE_PATH [DOWNLOADED CHECKPOINTED MODEL PATH]

```

### Testing:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=[PATH TO YOUR RoiSlowFast CODE] python tools/run_net.py --cfg [PATH TO YOUR RoiSlowFast CODE]/epic_config/ALPHA_4_BETA_8_ALL_BBOX_LOCAL_DATASET.yaml --init_method tcp://localhost:[LOCALHOST-NUMBER OF YOUR CHOICE] TRAIN.ENABLE False TEST.ENABLE True TEST.NUM_ENSEMBLE_VIEWS 8 NUM_GPUS [NUMBER OF GPUS AVAILABLE] OUTPUT_DIR [OUTPUT DIRECTORY PATH THAT HAS THE SAVED CHECKPOINTS] ENABLE_WANDB False EPICKITCHENS.ROI_BRANCH 3 EPICKITCHENS.VISUAL_DATA_DIR [EPIC-KITCHENS BASE DIRECTORY/EPIC-KITCHENS] EPICKITCHENS.ANNOTATIONS_DIR [DIRECTORY THAT CONTAINS DOWNLOADED ANNOTATIONS/epic-annotations] EPICKITCHENS.BBOX_ANNOTATIONS_DIR [DIRECTORY THAT CONTAINS DOWNLOADED BOUNDING BOX ANNOTATIONS/epic-annotations] EPICKITCHENS.TEST_LIST annotations_slowfast_val2.pkl EPICKITCHENS.TRAIN_LIST annotations_slowfast_train2.pkl EPICKITCHENS.VAL_LIST annotations_slowfast_val2.pkl EPICKITCHENS.SLOWFAST_PRETRAIN_CHECKPOINT_FILE_PATH [DOWNLOADED CHECKPOINTED MODEL PATH]

```
