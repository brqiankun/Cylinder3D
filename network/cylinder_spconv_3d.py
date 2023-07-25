# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn

import logging
logging.basicConfig(format='%(pathname)s->%(lineno)d: %(message)s', level=logging.INFO)
def stop_here():
    raise RuntimeError("🚀" * 5 + "-stop-" + "🚀" * 5)


REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):
    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_shape,
                 ):
        super().__init__()
        self.name = "cylinder_asym"

        self.cylinder_3d_generator = cylin_model

        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        # input: [124668, 9]点特征，包括和体素中心的距离，坐标，xy坐标，反射强度  (124668, 3) 每个点的体素坐标(int索引)
        logging.info(train_pt_fea_ten[0].shape)
        logging.info(train_vox_ten[0].shape)
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)  # 得到体素特征
        logging.info("coords.shape: {}\nfeatures_3d: {}\n".format(coords.shape, features_3d.shape))
        # stop_here()

        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)
        logging.info("spatial_features.shape: {}".format(spatial_features.shape))
        # stop_here()

        return spatial_features
