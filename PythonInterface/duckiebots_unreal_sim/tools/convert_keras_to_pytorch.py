
from duckiebots_unreal_sim.rcan_obs_preprocessor import RCANObsPreprocessor
import tensorflow as tf
if __name__ == '__main__':
    # rcan = RCANObsPreprocessor(checkpoint_path="/home/author1/Downloads/ckpt-71")
    rcan = RCANObsPreprocessor(checkpoint_path="/home/author1/Downloads/ckpt-91")

    # rcan.generator.save(filepath="/home/author1/Downloads/chkpt-94.h5", save_format="h5")
    # rcan.generator.save(filepath="/home/author1/Downloads/chkpt-94.keras")

    import tf2onnx

    spec = (tf.TensorSpec((None, 64, 64, 3), tf.float32, name="input"),)

    # model_proto, _ = tf2onnx.convert.from_keras(rcan.generator, input_signature=spec, output_path="/home/author1/Downloads/ckpt-71.onnx")
    model_proto, _ = tf2onnx.convert.from_keras(rcan.generator, input_signature=spec, output_path="/home/author1/Downloads/ckpt-91.onnx")

    output_names = [n.name for n in model_proto.graph.output]
    print(f"output names: {output_names}")
    print("done")

# mmconvert -sf keras -iw /home/author1/Downloads/chkpt-94.h5 -df pytorch -om /home/author1/Downloads/chkpt-94.pth

# mmconvert -sf keras -iw /home/author1/Downloads/chkpt-94.keras -df pytorch -om /home/author1/Downloads/chkpt-94.pth


# mmconvert -sf tensorflow -in imagenet_resnet_v2_152.ckpt.meta -iw imagenet_resnet_v2_152.ckpt --dstNodeName MMdnn_Output -df pytorch -om tf_resnet_to_pth.pth


# python -m tf2onnx.convert --saved-model /home/author1/Downloads/chkpt-94.keras --output /home/author1/Downloads/chkpt-94.onnx