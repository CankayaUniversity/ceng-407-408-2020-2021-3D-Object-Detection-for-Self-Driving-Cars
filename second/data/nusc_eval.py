'''
This script calculates mAP for given predictions using python's multiprocessing module, resulting in 10x faster calculation compared for standard for loop implementation.
`pred_file_path`: json file, predictions in global frame, in the format of:
predictions = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
    'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
    'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
    'name': 'car',
    'score': 0.3077029437237213
}]
`gt_file_path`: ground truth annotations in global frame, in the format of:
gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
}]
NOTICE both are lists of annotations in dicts.
'''

import fire
from pathlib import Path
import numpy as np
from numba import jit
from multiprocessing import Process

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from lyft_dataset_sdk.eval.detection.mAP_evaluation import *


def save_ap(gt, predictions, class_names, iou_threshold, output_dir):
    ''' computes average precisions (ap) for a given threshold, and saves the metrics in a temp file '''
    ap = get_average_precisions(gt, predictions, class_names, iou_threshold)
    metric = {c:ap[idx] for idx, c in enumerate(class_names)}
    summary_path = output_dir / f'metric_summary_{iou_threshold}.json'
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f)


def get_metric_overall_ap(iou_th_range, output_dir, class_names):
    ''' reads temp files and calculates overall metrics, returns:
    `metric`: a dict with key as iou thresholds and value as dicts of class and their respective APs,
    `overall_ap`: overall ap of each class
    '''

    metric = {}
    overall_ap = np.zeros(len(class_names))
    for iou_threshold in iou_th_range:
        summary_path = output_dir / f'metric_summary_{iou_threshold}.json'
        #import pdb; pdb.set_trace()
        with open(str(summary_path), 'r') as f:
            data = json.load(f) # type(data): dict
            metric[iou_threshold] = data
            overall_ap += np.array([data[c] for c in class_names])
        summary_path.unlink() # delete this file
    overall_ap /= len(iou_th_range)
    return metric, overall_ap


def eval_main(gt_file_path, pred_file_path, output_dir):
    '''
    Main function to compute mAP
    args:
    gt_file_path: json file path with ground truth annotations
    pred_file_path: json file path with predicted annotations
    output_dir: the final computed metrics are saved in this directory as a json file
    '''

    gt_path = Path(gt_file_path)
    pred_path = Path(pred_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(pred_path)) as f:
        predictions = json.load(f)

    with open(str(gt_path)) as f:
        gt = json.load(f)


    # order doesn't matter here
    class_names = ['animal', 'bicycle', 'bus', 'car', 'emergency_vehicle',
                    'motorcycle', 'other_vehicle', 'pedestrian', 'truck']
    '''
    class_names = [
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
    iou_th_range = np.linspace(0.5, 0.95, 10) # 0.5, 0.55, ..., 0.90, 0.95

    metric = {}

    def process_range(start, end):
        processes = []
        for iou_threshold in iou_th_range[start:end]:
            process = Process(target=save_ap,
                    args=(gt, predictions, class_names, iou_threshold, output_dir))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    # nusc's got large val set
    process_range(0, 10)
    #process_range(3, 6)
    #process_range(5, 10)


    # get overall metrics
    metric, overall_ap = get_metric_overall_ap(iou_th_range, output_dir, class_names)

    mAP = np.mean(overall_ap)
    #overall_ap_dict = {c: overall_ap[idx] for idx, c in enumerate(class_names)}
    #metric['overall'] = {c:overall_ap_dict[c] for c in sorted(class_names)}
    metric['overall'] = {c: overall_ap[idx] for idx, c in enumerate(class_names)}
    metric['mAP'] = mAP

    summary_path = Path(output_dir) / 'metric_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(metric, f, indent=4)


def eval_main_old(root_path, version, eval_version, res_path, eval_set, output_dir):
    #import pdb; pdb.set_trace()
    nusc = NuScenes(version=version, dataroot=str(root_path), verbose=False)

    cfg = config_factory(eval_version)
    nusc_eval = NuScenesEval(nusc, config=cfg, result_path=res_path, eval_set=eval_set,
                            output_dir=output_dir,
                            verbose=False)
    nusc_eval.main(render_curves=False)

if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    fire.Fire(eval_main)


'''
example command
python /home/keceli/3d-data/second.pytorch/second/data/nusc_eval.py --root_path="/home/keceli/3d-data/data/lyft/train" --version=v1.0-trainval --eval_version=cvpr_2019 --res_path="/home/keceli/second_test/all_fhd/results/step_5865/results_nusc.json" --eval_set=val --output_dir="/home/keceli/second_test/all_fhd/results/step_5865"
python /home/keceli/3d-data/second.pytorch/second/data/nusc_eval.py --gt_file_path="/home/keceli/3d-data/data/lyft/train/gt_data_val.json" --pred_file_path="/home/keceli/second_test/all.pp.lowa.config.4/results/step_304785/pred_data_val.json" --eval_set=val --output_dir=".tmp/"
# nuscenes
python /home/keceli/3d-data/second.pytorch/second/data/nusc_eval.py --gt_file_path="/home/keceli/3d-data/data/nuscenes/v1.0-trainval/gt_data_val.json" --pred_file_path="/home/keceli/second_test/all.pp.lowa.config.nusc/results/step_23447/pred_data_val.json"  --output_dir=".tmp/"
'''