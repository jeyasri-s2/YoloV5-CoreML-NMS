# Just run to define the decode function
import torch
import numpy as np

classLabels = ['Airplane', 'Pen',
            'Paper', 'Laptop', 'Toothbrush', 'Monitor', 'Person', 'Motorbike', 'Car',
            'Box', 'Photoframe', 'Engine', 'Book', 'Drawer', 'Coaster']
numberOfClassLabels = len(classLabels)
outputSize = numberOfClassLabels + 5

reverseModel = False



def postprocess():
    grids = []
    expanded_strides = []
    strides = [8, 16, 32]
    img_size = (640,640)
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    return grids, expanded_strides

def exponential(Z):
    return np.exp(Z)

def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((ny, nx, 2)).float()

def addExportLayerToCoreml(builder):
    '''
    Adds the YoloX export layer to the coreml model
    '''
    outputNames = [output.name for output in builder.spec.description.output]
    grids,expanded_strides = postprocess()

    for i, outputName in enumerate(outputNames):
        # outputs[..., :2]
        builder.add_slice(name=f"slice_coordinates_xy_{outputName}", input_name=outputName,
                          output_name=f"{outputName}_sliced_coordinates_xy", axis="width", start_index=0, end_index=2)

        # (outputs[..., :2] + grids)
        builder.add_bias(name=f"add_grids_to_xy{outputName}", input_name=
                                f"{outputName}_sliced_coordinates_xy", output_name=f"{outputName}_add_grids_to_xy", b=grids, shape_bias=grids.shape)

        #(outputs[..., :2] + grids) * expanded_strides
        print(expanded_strides.dtype, expanded_strides.shape)

        builder.add_load_constant_nd( name=f"expandedStrides_{outputName}", output_name=f"{outputName}_expandedStrides", constant_value=expanded_strides, shape=expanded_strides.shape)
        builder.add_elementwise(name=f"multiply_xy_by_expandedStrides_{outputName}", input_names=[
                                f"{outputName}_add_grids_to_xy", f"{outputName}_expandedStrides"], output_name=f"{outputName}_calculated_xy", mode="MULTIPLY")

        #outputs[..., 2:4]


        builder.add_slice(name=f"slice_coordinates_wh_{outputName}", input_name=outputName,
                          output_name=f"{outputName}_sliced_coordinates_wh", axis="width", start_index=2, end_index=4)
        # np.exp(outputs[..., 2:4])
        builder.add_unary(name=f"exp_wh{outputName}", input_name=f"{outputName}_sliced_coordinates_wh", output_name=f"{outputName}_exp_wh", mode='exp')

        builder.add_elementwise(name=f"multiply_wh_by_expandedStrides_{outputName}", input_names=[
                        f"{outputName}_exp_wh", f"{outputName}_expandedStrides"], output_name=f"{outputName}_calculated_wh", mode="MULTIPLY")


        builder.add_concat_nd(name=f"concat_coordinates_{outputName}", input_names=[
                              f"{outputName}_calculated_xy", f"{outputName}_calculated_wh"], output_name=f"{outputName}_raw_coordinates", axis=-1)
        builder.add_scale(name=f"normalize_coordinates_{outputName}", input_name=f"{outputName}_raw_coordinates",
                          output_name=f"{outputName}_raw_normalized_coordinates", W=torch.tensor([1 / 640]).numpy(), b=0, has_bias=False)

        ### Confidence calculation ###
        builder.add_slice(name=f"slice_object_confidence_{outputName}", input_name=outputName,
                          output_name=f"{outputName}_object_confidence", axis="width", start_index=4, end_index=5)
        builder.add_slice(name=f"slice_label_confidence_{outputName}", input_name=outputName,
                          output_name=f"{outputName}_label_confidence", axis="width", start_index=5, end_index=0)
        # confidence = object_confidence * label_confidence
        builder.add_multiply_broadcastable(name=f"multiply_object_label_confidence_{outputName}", input_names=[
                                           f"{outputName}_label_confidence", f"{outputName}_object_confidence"], output_name=f"{outputName}_raw_confidence")
        print("confidence calculated ")
        builder.add_flatten_to_2d(
            name=f"flatten_confidence_{outputName}", input_name=f"{outputName}_raw_confidence", output_name="raw_confidence", axis=-1)
        builder.add_flatten_to_2d(
            name=f"flatten_coordinates_{outputName}", input_name=f"{outputName}_raw_normalized_coordinates", output_name="raw_coordinates", axis=-1)


    builder.spec.description.output.add()
    builder.set_output(output_names=["raw_coordinates","raw_confidence"], output_dims=[(8400,4),(8400,numberOfClassLabels)])

# Just run to define the NMS function

def createNmsModelSpec(nnSpec):
    '''
    Create a coreml model with nms to filter the results of the model
    '''
    nmsSpec = ct.proto.Model_pb2.Model()
    nmsSpec.specificationVersion = 1

    # Define input and outputs of the model
    for i in range(2):
        nnOutput = nnSpec.description.output[i].SerializeToString()

        nmsSpec.description.input.add()
        nmsSpec.description.input[i].ParseFromString(nnOutput)

        nmsSpec.description.output.add()
        nmsSpec.description.output[i].ParseFromString(nnOutput)

    nmsSpec.description.output[0].name = "coordinates"
    nmsSpec.description.output[1].name = "confidence"

    # Define output shape of the model
    outputSizes = [4,numberOfClassLabels]
    for i in range(len(outputSizes)):
        maType = nmsSpec.description.output[i].type.multiArrayType
        # First dimension of both output is the number of boxes, which should be flexible
        maType.shapeRange.sizeRanges.add()
        maType.shapeRange.sizeRanges[0].lowerBound = 0
        maType.shapeRange.sizeRanges[0].upperBound = -1
        # Second dimension is fixed, for "confidence" it's the number of classes, for coordinates it's position (x, y) and size (w, h)
        maType.shapeRange.sizeRanges.add()
        maType.shapeRange.sizeRanges[1].lowerBound = outputSizes[i]
        maType.shapeRange.sizeRanges[1].upperBound = outputSizes[i]
        del maType.shape[:]

    # Define the model type non maximum supression
    nms = nmsSpec.nonMaximumSuppression

    nms.coordinatesInputFeatureName = "raw_coordinates"
    nms.confidenceInputFeatureName = "raw_confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.confidenceOutputFeatureName = "confidence"

    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    # Some good default values for the two additional inputs, can be overwritten when using the model
    nms.iouThreshold = 0.6
    nms.confidenceThreshold = 0.3
    nms.stringClassLabels.vector.extend(classLabels)

    return nmsSpec

# Just run to combine the model added decode and the NMS.
def combineModelsAndExport(builderSpec, nmsSpec, fileName, quantize=False):
    '''
    Combines the coreml model with export logic and the nms to one final model. Optionally save with different quantization (32, 16, 8) (Works only if on Mac Os)
    '''
    try:
        print(f'Combine CoreMl model with nms and export model')
        # Combine models to a single one
        pipeline = ct.models.pipeline.Pipeline(input_features=[("image", ct.models.datatypes.Array(3, 640, 640)),
                                                               ("iouThreshold", ct.models.datatypes.Double(
                                                               )),
                                                               ("confidenceThreshold", ct.models.datatypes.Double())], output_features=["confidence", "coordinates"])

        # Required version (>= ios13) in order for mns to work
        pipeline.spec.specificationVersion = 1

        pipeline.add_model(builderSpec)
        pipeline.add_model(nmsSpec)

        pipeline.spec.description.input[0].ParseFromString(
            builderSpec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(
            nmsSpec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(
            nmsSpec.description.output[1].SerializeToString())

        # Metadata for the model‚
        pipeline.spec.description.input[
            1].shortDescription = "(optional) IOU Threshold override (Default: 0.6)"
        pipeline.spec.description.input[
            2].shortDescription = "(optional) Confidence Threshold override (Default: 0.4)"
        pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"
        pipeline.spec.description.output[1].shortDescription = u"Boxes \xd7 Class confidence"

        pipeline.spec.description.metadata.versionString = "1.0"
        pipeline.spec.description.metadata.shortDescription = "yoloX"
        pipeline.spec.description.metadata.author = "Jeyasri"
        pipeline.spec.description.metadata.license = ""
        print("pipeline defined ")
        model = ct.models.MLModel(pipeline.spec)
        model.save(fileName)


        if quantize:
            fileName16 = fileName.replace(".mlmodel", "_16.mlmodel")
            modelFp16 = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=16)
            modelFp16.save(fileName16)

            fileName8 = fileName.replace(".mlmodel", "_8.mlmodel")
            modelFp8 = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=8)
            modelFp8.save(fileName8)

        print(f'CoreML export success, saved as {fileName}')
    except Exception as e:
        print(f'CoreML export failure: {e}')

if __name__ == "__main__":
# You need specify the path to your model that converted and saved in the same folder of your weight file.
     import coremltools as ct
     model_path = 'MyEngine-model/'
     model_file = 'yolox_engine.mlmodel'
     mlmodel = ct.models.MLModel(model_path+model_file)
  # Just run to get the mlmodel spec.
     spec = mlmodel.get_spec()
     builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
     # run the functions to add decode layer and NMS to the model.
     strides = [8, 16, 32]
     if reverseModel:
         strides.reverse()
     #featureMapDimensions = [640 // stride for stride in strides]
     addExportLayerToCoreml(builder)
     nmsSpec = createNmsModelSpec(builder.spec)
     combineModelsAndExport(builder.spec, nmsSpec, model_path+"yolox_engine_NMS.mlmodel") # The model will be saved in this path.

# Note: pip install "coremltools<6.3.0"
