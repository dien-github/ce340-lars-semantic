import os
from metrics import get_target_from_data, eval_results
from model import Model
from PIL import Image
import time
import numpy as np

if __name__ == "__main__":
    # ONLY CHANGE 3 LINES
    data_path = "D:\File_dai_hoc_UIT\Cac_do_an_UIT\LaRS_SemanticSegmentation\lars"  # path to dataset
    nc = 20  # number of class
    path_to_model = "./model.tflite"  # path to model
    #

    input_size = 320
    model = Model(model_path=path_to_model)
    model.prepare()

    image_path = data_path + "/images/"
    total_time = 0.0
    total_file = len(os.listdir(image_path))
    results = []
    targets = get_target_from_data(data_path)

    for fi in os.listdir(image_path):
        img = Image.open(image_path + "/" + fi).resize((input_size, input_size))
        labels, _ = targets[fi.rsplit(".", 1)[0]]

        start_time = time.time()
        preds = model.predict(img)
        stop_time = time.time()
        run_time = stop_time - start_time
        total_time += run_time
        results.append((preds, labels))

    FPS = total_file / total_time
    print("Average FPS: {:.3f}".format(FPS))
    normFPS = FPS / 10
    mp, mr, map50, map, f1 = eval_results(results, nc, input_size)
    score = 2 * normFPS * f1 / (normFPS + f1)
    print("Score: {:.3f}".format(score))
