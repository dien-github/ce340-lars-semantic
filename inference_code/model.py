import numpy as np
import tflite_runtime.interpreter as tflite

class Model(object):
    def __init__(self, model_path):
        #super.__init__()
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.BOX_COORD_NUM = 4

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        self.floating_model = self.input_details[0]["dtype"] == np.float32

        # NxHxWxC, H:1, W:2
        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]

        self.max_box_count = self.output_details[0]["shape"][2]

        self.class_count = self.output_details[0]["shape"][1] - self.BOX_COORD_NUM
        self.input_mean = 0.0
        self.input_std = 255.0
        self.keypoint_count = 0
        self.score_threshold = 0.6

    def prepare(self):
        return None

    def predict(self, image):
        input_data = np.expand_dims(image, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        results = np.squeeze(output_data).transpose()

        return results

