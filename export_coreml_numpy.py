#!/usr/bin/env python3


import argparse
import os
#from loguru import logger

import torch
from torch import nn

import sys
sys.path.append("../../yolox-code/yolox/")

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
import coremltools as ct

import coremltools.proto.FeatureTypes_pb2 as ft


#from onnx_coreml import convert

#import sys
#sys.path.append("../yolox-code/")

def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser


#@logger.catch
def main():
    args = make_parser().parse_args()
    print("args value: {}".format(args))
    PATH = '../network/Engine-model/'
    exp_file = PATH+args.exp_file
    exp = get_exp(exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = PATH+args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference

    print("loading checkpoint done.")
    #dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    inputs = torch.randn(1, 3, 640,640) #(2,3,640,640) worked !
    # initialize grids if they aren't initialized
    with torch.no_grad():
        out = model(inputs)


    """
    Image Size: 640 * 640
    """

    traced_model = torch.jit.trace(model, inputs,strict=False)
    #input_image = ct.ImageType(name="my_input", shape=(1, 3, 32, 32), scale=1/255)
    #input_image = ct.ImageType(name="inputs", shape=(1, 3, 640, 640), scale=1 / 255.0, bias=[0, 0, 0],channel_first=True)

    input_image = ct.ImageType(name="inputs", shape=(1,  3, 640, 640), scale=1 / 255,  bias=[0, 0, 0])
    #coreml_model = ct.convert(traced_model, inputs=[input_image])
    int8 = True
    half = False
    #bits, mode = (8,'kmeans_lut') if int8 else (16,'linear') if half else (32, None)


   # input_image = ct.ImageType(name="inputs", shape=(1, 3, 640, 640), scale=1 / 255.0, bias=[0,0,0])

   # coreml_model = ct.convert(traced_model, inputs=[input_image]) #worked!
    coreml_model = ct.convert(traced_model, inputs=[ct.TensorType(shape=inputs.shape)])

    spec = coreml_model.get_spec()
    #print("spec ",spec)
    ct.utils.rename_feature(spec, spec.description.output[0].name, "predictions")
    for shape in out.shape[1:]:
        spec.description.output[0].type.multiArrayType.shape.append(shape)

    #ct.utils.rename_feature(spec, "81", "my_output")
    coreml_model_updated = ct.models.MLModel(spec)
    output_name = PATH+args.output_name
    tmp_file = 'tmp.mlmodel'
    coreml_model_updated.save(tmp_file)


    print("generated coreml model named {}".format(tmp_file))

    spec = ct.utils.load_spec(tmp_file)

    input = spec.description.input[0]
    input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    input.type.imageType.height = 640
    input.type.imageType.width = 640

    output = spec.description.output[0]
    shape = [8400, 20] # dimension update as per nodel

    for s in shape:
        output.type.multiArrayType.shape.append(s)
    print(output)



    ct.utils.save_spec(spec, output_name)
    print("saved new model file @: ", output_name)




if __name__ == "__main__":
    main()


#python3 tools/export_coreml_numpy.py --output-name yolox_engine_np.mlmodel -f train_coffee_ego.py -c best_ckpt.pth
