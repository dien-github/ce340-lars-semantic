# onnx_tf convert -i "model.onnx" -o "model_tf"

import argparse
import tensorflow as tf
import os


def convert_saved_model_to_tflite(
    saved_model_dir, tflite_output_path, opset="TFLITE_BUILTINS"
):
    """
    Convert a TensorFlow SavedModel to TFLite format.

    Args:
        saved_model_dir (str): Path to the SavedModel directory.
        tflite_output_path (str): Path to save the .tflite file.
        opset (str, optional): TFLite opset to use ("TFLITE_BUILTINS" or "SELECT_TF_OPS").

    Returns:
        None
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if opset == "TFLITE_BUILTINS":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    elif opset == "SELECT_TF_OPS":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    tflite_model = converter.convert()
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Converted TFLite model saved to: {tflite_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TensorFlow SavedModel to TFLite."
    )
    parser.add_argument(
        "--saved_model_dir",
        required=True,
        help="Path to the TensorFlow SavedModel directory.",
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the .tflite file."
    )
    parser.add_argument(
        "--opset",
        default="TFLITE_BUILTINS",
        choices=["TFLITE_BUILTINS", "SELECT_TF_OPS"],
        help="TFLite opset to use (default: TFLITE_BUILTINS).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.saved_model_dir):
        print(f"SavedModel directory not found: {args.saved_model_dir}")
        return
    convert_saved_model_to_tflite(args.saved_model_dir, args.output, args.opset)


if __name__ == "__main__":
    main()
