import numpy as np
import pickle
from pathlib import Path
import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool

from alfred.dl.torch.common import device
from alfred.utils.log import init_logger
from loguru import logger as logging
import cv2
import sys
import time

from open3d import *
import pdb; pdb.set_trace()
from alfred.vis.pointcloud.pointcloud_vis import draw_pcs_open3d
from alfred.fusion.common import compute_3d_box_lidar_coords


init_logger()


class Second3DDector(object):

    def __init__(self, config_p, model_p, calib_data=None):
        self.config_p = config_p
        self.model_p = model_p
        self.calib_data = calib_data
        self._init_model()

    def _init_model(self):
        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(self.config_p, 'r') as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)
        
        self.input_cfg = self.config.eval_input_reader
        self.model_cfg = self.config.model.second
        config_tool.change_detection_range_v2(self.model_cfg, [-50, -50, 50, 50])
        logging.info('config loaded.')

        self.net = build_network(self.model_cfg).to(device).eval()
        self.net.load_state_dict(torch.load(self.model_p))
        
        self.target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator
        logging.info('network done, voxel done.')

        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2]//config_tool.get_downsample_factor(self.model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]

        self.anchors = self.target_assigner.generate_anchors(feature_map_size)['anchors']
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=device)
        self.anchors = self.anchors.view(1, -1, 7)
        logging.info('anchors generated.')

    @staticmethod
    def load_pc_from_file(pc_f):
        logging.info('loading pc from: {}'.format(pc_f))
        return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 5])

    def load_an_in_example_from_points(self, points):
        res = self.voxel_generator.generate(points, max_voxels=90000)
        voxels, coords, num_points = res['voxels'], res['coordinates'], res['num_points_per_voxel']
        coords = np.pad(coords, ((0,0), (1,0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
        coords = torch.tensor(coords, dtype=torch.int32, device=device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
        return {
            'anchors': self.anchors,
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coords,
        }

    def vis_lidar_prediction_on_img(self, img, boxes, scores, labels):
        for i in range(len(boxes)):
            score = scores[i]
            if score > 0.5:
                p = boxes[i]
                xyz = np.array([p[: 3]])
                c2d = lidar_pt_to_cam0_frame(xyz, self.calib_data)
                if c2d is not None:
                    cv2.circle(img, (int(c2d[0]), int(c2d[1])), 3, (0, 255, 255), -1)
                hwl = np.array([p[3: 6]])
                r_y = [p[6]]
                pts3d = compute_3d_box_lidar_coords(xyz, hwl, angles=r_y, origin=(0.5, 0.5, 0.5), axis=2)[0]
                pts3d_4 = np.ones((pts3d.shape[0], pts3d.shape[1]+1))
                pts3d_4[:,:-1] = pts3d
                _, pts2d = lidar_pts_to_cam0_frame(pts3d_4, self.calib_data)
                pts2d = pts2d[:2, :].T
                c = get_unique_color_by_id(labels[i])
                draw_3d_box(pts2d, img, c)
                if len(pts2d) > 4:
                    cv2.putText(img, '{0} {1:.1f}'.format(self.label_map[labels[i]], score),
                        (int(pts2d[2][0]), int(pts2d[2][1])), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255))
        return img

    def predict_on_nucenes_local_file(self, v_p):
        tic = time.time()
        points = self.load_pc_from_file(v_p)[:, :4]
        print('points shape: ', points.shape)
        example = self.load_an_in_example_from_points(points)
        pred = self.net(example)[0]
        box3d = pred['box3d_lidar'].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["label_preds"].detach().cpu().numpy()

        idx = np.where(scores > 0.11)[0]
        box3d = box3d[idx, :]
        labels = np.take(labels, idx)
        scores = np.take(scores, idx)

        # show points first
        geometries = []
        pcs = np.array(points[:,:3])
        pcobj = PointCloud()
        pcobj.points = Vector3dVector(pcs)
        geometries.append(pcobj)
        # try getting 3d boxes coordinates
        for p in box3d:
            xyz = np.array([p[: 3]])
            hwl = np.array([p[3: 6]])
            r_y = [p[6]]
            pts3d = compute_3d_box_lidar_coords(xyz, hwl, angles=r_y, origin=(0.5, 0.5, 0.5), axis=2)[0]
            points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                [0,0,1],[1,0,1],[0,1,1],[1,1,1]]
            print(pts3d, points)
            lines = [[0,1],[1,2],[2,3],[3,0],
                [4,5],[5,6],[6,7],[7,4],
                [0,4],[1,5],[2,6],[3,7]]
            colors = [[1, 0, 1] for i in range(len(lines))]
            line_set = LineSet()
            line_set.points = Vector3dVector(pts3d)
            line_set.lines = Vector2iVector(lines)
            line_set.colors = Vector3dVector(colors)
            geometries.append(line_set)

        draw_pcs_open3d(geometries)

    def predict_on_points(self, points):
        example = self.load_an_in_example_from_points(points)
        pred = self.net(example)[0]
        boxes_lidar = pred['box3d_lidar'].detach().cpu().numpy()
        return boxes_lidar


if __name__ == "__main__":

    config_p = 'second/configs/nuscenes/all.fhd.config'
    model_p = '/home/ags/second_test/all_fhd/voxelnet-29369.tckpt'
    lidar_file = '/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/data/lyft/test/'
    lidar_file += 'lidar/host-a004_lidar1_1231810077401389686.bin'
    detector = Second3DDector(config_p, model_p)

    detector.predict_on_nucenes_local_file(lidar_file)