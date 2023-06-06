import torch
from torch_scatter import scatter_max
import spconv.pytorch as spconv
import numpy as np

src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])

out , argmax = scatter_max(src, index, dim=-1)

print(out)

print(argmax)
binpath = "/home/br/program/cylinder3d/work/infer_test/velodyne/000001.bin"
raw_data = np.fromfile(binpath, dtype=np.float32)
print(raw_data.shape)
print("raw_data: {}".format(raw_data))
raw_data_reshape = raw_data.reshape((-1, 4))
print(raw_data_reshape.shape)
print("raw_data_reshape: {}".format(raw_data_reshape))
print("raw_data_reshape[:, :0]: {}".format(raw_data_reshape[:, :0]))
print("raw_data_reshape[:, 0:1]: {}".format(raw_data_reshape[:, 0:1]))
assert raw_data_reshape[:, 0] == raw_data_reshape[:, 0:1]
annotated_data = np.expand_dims(np.zeros_like(raw_data_reshape[:, 0], dtype=int), axis=1)

data_tuple = (raw_data_reshape[:, :3], annotated_data.astype(np.uint8))
print(data_tuple)