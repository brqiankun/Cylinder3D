import onnx
import torch
import os
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint

import logging
logging.basicConfig(format='%(pathname)s->%(lineno)d: %(message)s', level=logging.INFO)

pytorch_device = torch.device('cuda:0')
curr_dir = os.path.realpath(".")

def convert_onnx(model):

    # set the model to inference model
    model.eval()
    # cylinder3d输入是含有tensor的list
    # predict_labels = my_model(demo_pt_fea_ten, demo_grid_ten, demo_batch_size)
    # create a dummy input tensor
    dummy_input_pt_fea_ten = []
    dummy_input_grid_ten = []
    dummy_input_pt_fea_ten.append(torch.randn([60000, 9]).to(pytorch_device))
    dummy_input_grid_ten.append(torch.randn([60000, 3]).to(pytorch_device))
    batch_sz = 1
    # export the model
    output = my_model(dummy_input_pt_fea_ten, dummy_input_grid_ten, batch_sz)
    logging.info("output.shape: {}".format(output.shape))
    predict_labels = torch.argmax(output, dim=1)
    predict_labels = predict_labels.cpu().detach().numpy()
    logging.info("predict_labels.shape: {}".format(predict_labels.shape))
    torch.onnx.export(model,      # model being run
        (dummy_input_pt_fea_ten, dummy_input_grid_ten, batch_sz),              # model input (or a tuple for multiple input)
        os.path.join(curr_dir, "cylinder3d_pretrained.onnx"),             # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=11,         # the ONNX version to export the model to
        do_constant_folding=True, # whether tot execute constant folding for optimization
        input_names=["input_pt_fea_ten", "input_grid_ten", "batch_sz"],          # the model's input name
        output_names=["output"],         # the model's output name
        dynamic_axes=None)         # variable length axes  ????

    logging.info("model has been converted to ONNX")


if __name__ == "__main__":
    config_path = "./config/semantickitti.yaml"
    configs = load_config_data(config_path)

    model_config = configs['model_params']
    train_hypers = configs['train_params']
    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    model_load_path = train_hypers['model_load_path']
    model_config = configs['model_params']
    my_model = model_builder.build(model_config).to(pytorch_device)
        # raise RuntimeError("model build done")
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)
    logging.info("--------------model load done-------------")
    logging.info("my_model:\n{}".format(my_model))
    # test with image

    # convert to onnx
    convert_onnx(my_model)
