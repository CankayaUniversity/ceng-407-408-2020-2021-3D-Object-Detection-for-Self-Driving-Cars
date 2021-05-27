'''
This script takes in raw detections (net() outputs, in lidar frame) [use submission.ipynb] and ground truth boxes global frame to calculate lyft's mAP metric
'''

import time
import pickle
from math import cos, sin, pi
from copy import deepcopy
from tqdm import tqdm
from second.data.nusc_eval import *
from lyft_dataset_sdk.lyftdataset import *

def thresholded_pred(pred, threshold):
    pred = deepcopy(pred)
    box3d = pred["box3d_lidar"]#.detach().cpu().numpy()
    scores = pred["scores"]#.detach().cpu().numpy()
    labels = pred["label_preds"]#.detach().cpu().numpy()
    idx = np.where(scores > threshold)[0]
    # filter low score ones
    box3d = box3d[idx, :]
    # label is one-dim
    labels = np.take(labels, idx)
    scores = np.take(scores, idx)
    pred['box3d_lidar'] = box3d
    pred['scores'] = scores
    pred['label_preds'] = labels
    return pred

def to_glb(box, info):
    # lidar -> ego -> global
    # info should belong to exact same element in `gt` dict
    box.rotate(Quaternion(info['lidar2ego_rotation']))
    box.translate(np.array(info['lidar2ego_translation']))
    '''
    # filter det in ego.
    cls_range_map = eval_detection_configs[eval_version]["class_range"]
    radius = np.linalg.norm(box.center[:2], 2)
    det_range = cls_range_map[classes[box.label]]
    if radius > det_range:
        continue
    '''
    box.rotate(Quaternion(info['ego2global_rotation']))
    box.translate(np.array(info['ego2global_translation']))
    return box


def get_pred_glb(pred, sample_token, classes, token2info):
    boxes_lidar = pred["box3d_lidar"]
    boxes_class = pred["label_preds"]
    scores = pred['scores']
    preds_classes = [classes[x] for x in boxes_class]
    box_centers = boxes_lidar[:, :3]
    box_yaws = boxes_lidar[:, -1]
    box_wlh = boxes_lidar[:, 3:6]
    info = token2info[sample_token] # a `sample` token
    boxes = []
    for idx in range(len(boxes_lidar)):
        translation = box_centers[idx]
        yaw = - box_yaws[idx] - pi/2 # second to lyft format
        size = box_wlh[idx]
        name = preds_classes[idx]
        detection_score = scores[idx]
        quat = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        box = Box(
            center=box_centers[idx],
            size=size,
            orientation=quat,
            score=detection_score,
            name=name,
            token=sample_token
        )
        box = to_glb(box, info)
        boxes.append(box)
    return boxes



def serialize(box):
    return {
        'sample_token': box.token,
        'translation': list(box.center),
        'size': list(box.wlh),
        'rotation': list(box.orientation.elements),
        'name': box.name,
        'score': box.score
    }


def main(det_file, phase='train'):
    '''
    det_file: pkl file containing raw detections (just out of net(), in lidar's frame)
    flow:
    1. get the detections to global from using transformation matrices stored in info pickle files, and get the serialized repr of the boxes in a list.
    2. read the ground truth global frame serialized boxes.
    3. use eval functions in nusc_eval.py to calculate mAP, using multiprocessing.
    4. save the metrics to notebooks/.tmp/ directory
    '''

    t0 = time.time()
    output_dir = Path('notebooks/.tmp')
    det_file = Path(det_file)
    print(f'Phase: {phase}')
    root = Path('/home/keceli/3d-data')
    gt_file = root / f'data/lyft/train/gt_data_{phase}.json'
    info_path = root / f'data/lyft/train/infos_{phase}.pkl'

    '''
    gt_file = root / f'data/nuscenes/v1.0-trainval/gt_data_{phase}.json'
    info_path = root / f'data/nuscenes/v1.0-trainval/infos_{phase}.pkl'
    '''

    with open(str(info_path), 'rb') as f:
        infos = pickle.load(f)

    token2info = {}
    for info in infos['infos']:
        token2info[info['token']] = info

    del infos

    with open(str(det_file), 'rb') as f:
        detections = pickle.load(f)#[:2000]

    with open(str(gt_file)) as f:
        gboxes = json.load(f)

    classes = ['car', 'bicycle', 'animal', 'bus',
            'emergency_vehicle', 'other_vehicle',
            'motorcycle', 'pedestrian', 'truck']
    '''
    classes = [
         'car',
         'bicycle',
         'bus',
         'construction_vehicle',
         'motorcycle',
         'pedestrian',
         'traffic_cone',
         'trailer',
         'truck'
         'barrier'
    ]
    '''
    # order is imp., raw detections' labels are indices acc to above list
    # order is dicided by the order of class definitions in config file

    threshold = 0.2
    pboxes = []
    print('Processing raw predictions..')
    for idx, pred in enumerate(tqdm(detections)):
        pred = thresholded_pred(pred, threshold)
        token = pred['metadata']['token']
        boxes = get_pred_glb(pred, token, classes, token2info)
        pboxes.extend([serialize(box) for box in boxes])

    del detections

    classes = list(sorted(classes)) # now classes can be sorted.

    print('Done')
    print('Starting mAP computation..')
    # now we have all serialized pred boxes in pboxes and gt boxes in gboxes

    iou_th_range = np.linspace(0.5, 0.95, 10) # 0.5, 0.55, ..., 0.90, 0.95
    def process_range(start, end):
        processes = []
        for iou_threshold in iou_th_range[start:end]:
            process = Process(target=save_ap,
                    args=(gboxes, pboxes, classes, iou_threshold, output_dir))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
    '''
    if phase == 'train':
        process_range(0, 10)
        print('got half way through,..')
        process_range(5, 10)
    else:
        process_range(0, 10)
    '''

    # nusc's got large val set
    process_range(0, 10)
    #process_range(2, 4)
    #process_range(4, 6)
    #process_range(6, 8)
    #process_range(8, 10)

    metric, overall_ap = get_metric_overall_ap(iou_th_range, output_dir, classes)

    mAP = np.mean(overall_ap)
    metric['overall'] = {c: overall_ap[idx] for idx, c in enumerate(classes)}
    metric['mAP'] = mAP

    summary_path = output_dir / f'{det_file.name}.json'
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f, indent=4)

    print(f'{json.dumps(metric, indent=2)}')

    diff = time.time() - t0
    print("Total time taken : %02d:%02d" % (diff // 60, diff % 60))


if __name__ == "__main__":
    fire.Fire(main)


'''
 python eval.py --det_file /home/keceli/second_test/all.pp.mhead.config.13/detections/voxelnet-58650_val.pkl --phase val
'''