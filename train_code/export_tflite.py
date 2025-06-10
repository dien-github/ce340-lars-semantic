# onnx_tf convert -i "model.onnx" -o "model_tf"

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("best_model_20250604_1_pruned_tf")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
with open("20250604_1_pruned.tflite", "wb") as f:
    f.write(tflite_model)
