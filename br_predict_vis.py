import numpy as np
import yaml
import open3d
from pathlib import Path

import logging
logging.basicConfig(format='%(pathname)s->%(lineno)d: %(message)s', level=logging.INFO)
def stop_here():
    raise RuntimeError("🚀" * 5 + "-stop-" + "🚀" * 5)

 
# points_dir = Path('/media/br/kitti_odometry/dataset/sequences/11/velodyne') # path to .bin data
# points_dir = Path('/mnt/d/bairui/brfile/dataset/SemanticKitti/dataset/sequences/11/velodyne') # path to .bin data
points_dir = Path('/media/br/kitti_odometry/dataset/sequences/00/velodyne') # path to .bin data
# label_dir = Path('/home/bairui/program/2dpass/checkpoint/submit_2023_07_07/sequences/11/predictions') # path to .label data
label_dir = Path('/home/br/program/cylinder3d/work/infer_test/labels') # path to .label data

# label_filter = [40, 48, 70, 72]    # object's label which you wan't to show
label_filter = []
# with open('/home/bairui/program/2dpass/config/label_mapping/semantic-kitti.yaml', 'r') as stream: # label_mapping configuration file
with open('/home/br/program/cylinder3d/config/label_mapping/semantic-kitti.yaml', 'r') as stream: # label_mapping configuration file
    label_mapping = yaml.safe_load(stream)
    color_map_dict = label_mapping['color_map']
    print("\ncolor_map_dict: {}\n\n".format(color_map_dict))


def get_rgb_list(_label):
    c = color_map_dict[_label]
    return np.array((c[2], c[1], c[0]))


def draw_pc(pc_xyzrgb):
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
    pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)

    def custom_draw_geometry_with_key_callback(pcd):
        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            opt.point_size = 1
            return False

        key_to_callback = {}
        key_to_callback[ord("K")] = change_background_to_black
        open3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

    custom_draw_geometry_with_key_callback(pc)


def concate_color(_points, _label):
    color = np.zeros((_points.shape[0], 3))
    label_id = np.unique(_label)      # [ 0 10 30 31 40 44 48 49 50 51 70 72 80 81] 输出label直接对应semantic-kitti.yaml的labels
    logging.info("label_id: {}".format(label_id))
    for cls in label_id:
        if label_filter.__len__() == 0:
            color[_label == cls] = get_rgb_list(cls)
        elif label_filter.count(cls) == 0:
            color[_label == cls] = get_rgb_list(cls)
    _points = np.concatenate([_points, color], axis=1)
    return _points


for it in label_dir.iterdir():
    label_file = it
    points_file = points_dir / (str(it.stem) + '.bin')
    label = np.fromfile(label_file, dtype=np.uint32)
    points = np.fromfile(points_file, dtype=np.float32).reshape((-1, 4))[:, 0:3]
    colorful_points = concate_color(points, label)
    print("🚀label.shape: {}\npoints.shape: {}\ncolorful_points.shape: {}".format(label.shape, points.shape, colorful_points.shape))
    draw_pc(colorful_points)