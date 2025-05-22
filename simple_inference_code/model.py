import numpy as np
import tflite_micro_runtime.interpreter as tflite

from nms import non_max_suppression_yolov8

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

        boxes = []
        for i in range(self.max_box_count):
            raw_box = results[i]
            center_x = raw_box[0]
            center_y = raw_box[1]
            w = raw_box[2]
            h = raw_box[3]
            class_scores = raw_box[self.BOX_COORD_NUM:]
            for index, score in enumerate(class_scores):
                if (score > self.score_threshold):
                    boxes.append([center_x, center_y, w, h, score, index])

        clean_boxes = non_max_suppression_yolov8(boxes, self.class_count, self.keypoint_count)

        #return clean_boxes
        results = []

        for box in clean_boxes:
            center_x = box[0] * self.input_width
            center_y = box[1] * self.input_height
            w = box[2] * self.input_width
            h = box[3] * self.input_height
            half_w = w / 2
            half_h = h / 2
            left_x = int(center_x - half_w)
            right_x = int(center_x + half_w)
            top_y = int(center_y - half_h)
            bottom_y = int(center_y + half_h)
            score = box[4]
            class_index = box[5]
            results.append([left_x, top_y, right_x, bottom_y, score, class_index])

        return results