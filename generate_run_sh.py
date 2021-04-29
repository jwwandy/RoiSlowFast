import argparse
import os 

'''
python generate_run_sh.py --annotation_base /raid/xiaoyuz1/EPIC/epic-annotations --config_fpath test_config.yaml --output_dir_base /raid/xiaoyuz1/EPIC/slowfast_experiment --min_len 1 --max_len 64 --anno_format 'annotations_{}.pkl' --pid P02 --view 1 --output_path /raid/xiaoyuz1/EPIC/run_slowfast_sh/P02_view1.sh 

'''

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--output_path', type=str)
parser.add_argument('--annotation_base', type=str)
parser.add_argument('--config_fpath', type=str)
parser.add_argument('--output_dir_base', type=str)
parser.add_argument('--min_len', type=int)
parser.add_argument('--max_len', type=int)
parser.add_argument('--anno_format', type=str)
parser.add_argument('--view', type=int, default=10)

parser.add_argument('--pid', type=str, default='')
parser.add_argument('--vid', type=str, default='')

args = parser.parse_args()

vids = set()
if args.vid != '':
    vids.add(args.vid)
else:
    pids = set()
    if args.pid != '':
        pids.add(args.pid)
    else:
        for f in os.listdir(args.annotation_base):
            if f.split('.')[-1] == 'pkl':
                name = f.split('.')[0]
                vid = name[name.index('P'):]
                pids.add(vid.split('_')[0])
    pids = list(pids)
    pids.sort()

    for f in os.listdir(args.annotation_base):
        if f.split('.')[-1] == 'pkl':
            name = f.split('.')[0]
            vid = name[name.index('P'):]
            if vid.split('_')[0] in pids:
                vids.add(vid)
vids = list(vids)
python_path = 'CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/home/xiaoyuz1/epic-slow-fast python tools/run_net.py'
test_config = '--cfg {}'.format(args.config_fpath)
gpu = 'NUM_GPUS 1'
visual_data_dir = 'EPICKITCHENS.VISUAL_DATA_DIR /raid/xiaoyuz1/EPIC/EPIC-KITCHENS'
anno_dir = 'EPICKITCHENS.ANNOTATIONS_DIR /raid/xiaoyuz1/EPIC/epic-annotations'
enable = 'TRAIN.ENABLE False TEST.ENABLE True'
ckpt = 'TEST.CHECKPOINT_FILE_PATH /raid/xiaoyuz1/EPIC/SlowFast.pyth'
view = 'TEST.NUM_ENSEMBLE_VIEWS {}'.format(args.view)

commands = []
for vid in vids:
    l = [python_path, test_config, gpu, visual_data_dir, anno_dir, enable, ckpt, view]

    output = 'OUTPUT_DIR {}'.format(os.path.join(args.output_dir_base, '{}_{}_{}_{}'.format(vid, args.min_len, args.max_len, args.view)))
    fpath = args.anno_format.format(vid)
    train_list = 'EPICKITCHENS.TRAIN_LIST {}'.format(fpath)
    val_list = 'EPICKITCHENS.VAL_LIST {}'.format(fpath)
    test_list = 'EPICKITCHENS.TEST_LIST {}'.format(fpath)

    l += [output, train_list, val_list, test_list]

    l.append('EPICKITCHENS.SEGMENT_MIN_LENGTH {}'.format(args.min_len))
    l.append('EPICKITCHENS.SEGMENT_MAX_LENGTH {}'.format(args.max_len))
    commands.append(' '.join(l))

with open(args.output_path, 'w+') as filehandle:
    for com in commands:
        filehandle.write('{}\n'.format(com))