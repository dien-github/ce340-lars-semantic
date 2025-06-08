#Adapt from https://github.com/ultralytics/yolov5/blob/master/utils/metrics.py
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics."""

# import math
# import warnings
from pathlib import Path

#import matplotlib.pyplot as plt
import numpy as np
import threading
import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image
import cv2

def threaded(func):
    """Decorator @threaded to run a function in a separate thread, returning the thread instance."""

    def wrapper(*args, **kwargs):
        """Runs the decorated function in a separate daemon thread and returns the thread instance."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def fitness(x):
    """Calculates fitness of a model using weighted sum of metrics P, R, mAP@0.5, mAP@0.5:0.95."""
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    """Applies box filter smoothing to array `y` with fraction `f`, yielding a smoothed array."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Compute the average precision, given the recall and precision curves.

    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    # names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    # names = dict(enumerate(names))  # to dict
    # if plot:
    #     plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
    #     plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1")
    #     plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision")
    #     plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall")

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def bbox_ioa(box1, box2, eps=1e-7):
    """
    Returns the intersection over box2 area given box1, box2.

    Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def get_target_from_data(data_path, dataset_name, size):
    #task: detect (for object detection), instance (for instance segmentation), semantic (for semantic segmentation)
    task = ""
    #detect: "idd_fgvd", "rsud20k", "grazpedwri-dx", "seadronessee object detection v2", "svrdd", 
    #instance: "btxrd", "rip current", "trashcan",
    #semantic: "lars", "rescuenet", "loveda"
    if dataset_name == "rsud20k" or dataset_name == "grazpedwri-dx": #YOLO format
        test_folder_name = "test"
        if dataset_name == "grazpedwri-dx":
            test_folder_name = "valid"
        images_path = data_path + f'{os.sep}images{os.sep}{test_folder_name}{os.sep}'
        paths = []
        for fi in os.listdir(images_path):
            paths.append(images_path + f"{os.sep}" + fi)
        lb_paths = img2label_paths(paths)
        lb_dict = dict()
        for lb_file in lb_paths:
            label, _ = get_label(lb_file)
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = label

        return lb_dict

    elif dataset_name == "svrdd": #YOLO format with custom folder
        paths = [data_path + f'{os.sep}' + line.rstrip().replace("\\",f'{os.sep}') for line in open(data_path + f'{os.sep}test.txt')]
        lb_paths = img2label_paths(paths)
        lb_dict = dict()
        for lb_file in lb_paths:
            label, _ = get_label(lb_file)
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = label

        return lb_dict

    elif dataset_name == "idd_fgvd": #VOC format
        images_path = data_path + f'{os.sep}test{os.sep}images{os.sep}'
        paths = []
        class_lst = [line.rstrip() for line in open(data_path + f'{os.sep}class_names.txt')] 
        class_dict = dict()
        lb_dict = dict()
        for line in class_lst:
            lb, name = line.split(":")
            class_dict[name.strip()] = int(lb.strip())

        for fi in os.listdir(images_path):
            paths.append(images_path + f"{os.sep}" + fi)

        lb_paths = img2label_paths(paths, lb_dir="annos", lb_ext=".xml")
        for lb_file in lb_paths:
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = get_label_voc(lb_file, class_dict)

        return lb_dict

    elif dataset_name == "seadronessee object detection v2": #COCO format
        #https://github.com/ultralytics/JSON2YOLO/blob/main/general_json2yolo.py
        json_file = data_path + f'{os.sep}instances_val.json'
        class_lst = [line.rstrip() for line in open(data_path + f'{os.sep}class_names.txt')] 
        class_dict = dict()
        for line in class_lst:
            lb, name = line.split(":")
            class_dict[name.strip()] = int(lb.strip())

        images_path = data_path + f'{os.sep}val{os.sep}'
        paths = []
        for fi in os.listdir(images_path):
            paths.append(fi)

        lb_dict = dict()

        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        categories = data['categories']
        # print(categories)
        # print(class_dict)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        for img_id, anns in imgToAnns.items():
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            if f not in paths:
                continue

            bboxes = []
            # segments = []
            for ann in anns:
                # if ann["iscrowd"]:
                #     continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                #print(ann["category_id"])

                cls = ann["category_id"] #class_dict[categories[ann["category_id"]]]
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    
            bboxes = np.array(bboxes, dtype=np.float32)

            lb_dict[img["file_name"].rsplit(".",1)[0]] = bboxes

        return lb_dict

    elif dataset_name == "btxrd": #mixed format, JSON
        images_path = data_path + f'{os.sep}val{os.sep}'
        paths = []
        class_lst = [line.rstrip() for line in open(data_path + f'{os.sep}class_names.txt')] 
        class_dict = dict()
        lb_dict = dict()
        for line in class_lst:
            lb, name = line.split(":")
            class_dict[name.strip()] = int(lb.strip())

        for fi in os.listdir(images_path):
            paths.append(images_path + f"{os.sep}" + fi)

        lb_paths = img2label_paths(paths, img_dir="val", lb_dir="annotations", lb_ext=".json")

        lb_dict = dict()

        for lb_file in lb_paths:
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = get_label_json(lb_file, class_dict)

        return lb_dict

    elif dataset_name == "rip current": 
        lb_paths = []
        lb_dir_path = data_path + f'{os.sep}labels_edit{os.sep}'
        for scene in os.listdir(lb_dir_path):
            scene_dir_path = lb_dir_path + f'{os.sep}' + scene
            for fi in os.listdir(scene_dir_path):
                lb_paths.append(scene_dir_path + f"{os.sep}" + fi)

        lb_dict = dict()
        for lb_file in lb_paths:
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = get_label(lb_file)

        return lb_dict
    
    elif dataset_name == "carparts": 
        lb_paths = []
        lb_dir_path = data_path + f'{os.sep}labels'
        
        for fi in os.listdir(lb_dir_path):
            lb_paths.append(lb_dir_path + f"{os.sep}" + fi)

        lb_dict = dict()
        for lb_file in lb_paths:
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = get_label(lb_file)

        return lb_dict

    elif dataset_name == "trashcan": #COCO format
        #https://github.com/ultralytics/JSON2YOLO/blob/main/general_json2yolo.py
        json_file = data_path + f'{os.sep}instances_val_trashcan.json'
        class_lst = [line.rstrip() for line in open(data_path + f'{os.sep}class_names.txt')] 
        class_dict = dict()
        for line in class_lst:
            lb, name = line.split(":")
            class_dict[name.strip()] = int(lb.strip())

        images_path = data_path + f'{os.sep}val{os.sep}'
        paths = []
        for fi in os.listdir(images_path):
            paths.append(fi)

        lb_dict = dict()

        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        # categories = data['categories']
        # print(categories)
        # print(class_dict)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        for img_id, anns in imgToAnns.items():
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            if f not in paths:
                continue

            bboxes = []
            segments = []
            for ann in anns:

                cls = ann["category_id"] - 1 
                # box = [cls] + box.tolist()
                # if box not in bboxes:
                #     bboxes.append(box)

                s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                s = np.array(s).reshape(-1, 2) / np.array([w, h])
                box = np.concatenate((np.array([cls], dtype=np.float32), segments2boxes(s)[0]))
                #s = [cls] + s
                #if box not in bboxes:
                bboxes.append(box)
                #if s not in segments:
                segments.append(s)
            
            bboxes = np.array(bboxes, dtype=np.float32)

            lb_dict[img["file_name"].rsplit(".",1)[0]] = (bboxes, segments)

        return lb_dict

    elif dataset_name == "lars": #mask
        images_path = data_path + f'{os.sep}images{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        lb_paths = img2label_paths(paths, img_dir="images", lb_dir="semantic_masks", lb_ext=".png")

        lb_dict = dict()
        for lb_file in lb_paths:
            img = Image.open(lb_file)
            mask = np.array(img.resize((size, size)))
            mask = np.stack([mask==0, mask==1, mask==2], axis=-1).astype(np.float32)
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = mask

        return lb_dict

    elif dataset_name == "rescuenet": #mask
        images_path = data_path + f'{os.sep}segmentation-validationset{os.sep}val-org-img{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        lb_paths = img2label_paths(paths, img_dir="val-org-img", lb_dir="val-label-img", lb_ext=".png")

        lb_dict = dict()
        for lb_file in lb_paths:
            img = Image.open(lb_file.replace(".png", "_lab.png"))
            mask = np.array(img.resize((size, size)))
            mask = np.stack([mask==0, mask==1, mask==2, mask==3, mask==4, mask==5, mask==6, mask==7, mask==8, mask==9, mask==10], axis=-1).astype(np.float32)
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = mask

        return lb_dict

    elif dataset_name == "loveda": #mask
        images_path = data_path + f'{os.sep}Val{os.sep}Rural{os.sep}images_png{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        lb_paths = img2label_paths(paths, img_dir="images_png", lb_dir="masks_png", lb_ext=".png")

        images_path = data_path + f'{os.sep}Val{os.sep}Urban{os.sep}images_png{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        lb_paths1 = img2label_paths(paths, img_dir="images_png", lb_dir="masks_png", lb_ext=".png")
        lb_paths.extend(lb_paths1)

        lb_dict = dict()
        for lb_file in lb_paths:
            img = Image.open(lb_file)
            mask = np.array(img.resize((size, size)))
            mask = np.stack([mask==0, mask==1, mask==2, mask==3, mask==4, mask==5, mask==6, mask==7], axis=-1).astype(np.float32)
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = mask

        return lb_dict

    elif dataset_name == 'camvid':
        images_path = data_path + f'{os.sep}val{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        lb_paths = img2label_paths(paths, img_dir="val", lb_dir="val_labels", lb_ext=".png")

        class_lst = [line.rstrip() for line in open(data_path + f'{os.sep}class_dict.csv')] 

        color_map = []

        for line in class_lst[1:]:
            parts = line.split(',')
            color_map.append([int(parts[1].strip()), int(parts[2].strip()), int(parts[3].strip())])

        lb_dict = dict()
        for lb_file in lb_paths:
            img = Image.open(lb_file.replace('.png', '_L.png'))
            mask = np.array(img.resize((size, size)))
            mask = process_mask(mask, color_map)
            lb_dict[lb_file.rsplit(f"{os.sep}")[-1].rsplit(".",1)[0]] = mask

        return lb_dict

def get_image_paths(data_path, dataset_name):
    #detect: "idd_fgvd", "rsud20k", "grazpedwri-dx", "seadronessee object detection v2", "svrdd", 
    #instance: "btxrd", "rip current", "trashcan",
    #semantic: "lars", "rescuenet", "loveda"
    if dataset_name == "rsud20k" or dataset_name == "grazpedwri-dx": #YOLO format
        test_folder_name = "test"
        if dataset_name == "grazpedwri-dx":
            test_folder_name = "valid"
        images_path = data_path + f'{os.sep}images{os.sep}{test_folder_name}{os.sep}'
        paths = []
        for fi in os.listdir(images_path):
            paths.append(images_path + f"{os.sep}" + fi)

        return paths

    elif dataset_name == "svrdd": #YOLO format with custom folder
        paths = [data_path + f'{os.sep}' + line.rstrip().replace("\\",f'{os.sep}') for line in open(data_path + f'{os.sep}test.txt')]
        return paths

    elif dataset_name == "idd_fgvd": #VOC format
        images_path = data_path + f'{os.sep}test{os.sep}images{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + f"{os.sep}" + fi)

        return paths

    elif dataset_name == "seadronessee object detection v2": #COCO format
        #https://github.com/ultralytics/JSON2YOLO/blob/main/general_json2yolo.py
        images_path = data_path + f'{os.sep}val{os.sep}'
        paths = []
        for fi in os.listdir(images_path):
            paths.append(images_path + f"{os.sep}" + fi)

        return paths

    elif dataset_name == "btxrd": #mixed format, JSON
        images_path = data_path + f'{os.sep}val{os.sep}'
        paths = []
        for fi in os.listdir(images_path):
            paths.append(images_path + f"{os.sep}" + fi)

        return paths
    
    elif dataset_name == 'carparts':
        images_path = data_path + f'{os.sep}images{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        return paths

    elif dataset_name == "rip current": 
        paths = []
        dir_path = data_path + f'{os.sep}frames (sampled){os.sep}'
        for scene in os.listdir(dir_path):
            scene_dir_path = dir_path + f'{os.sep}' + scene
            for fi in os.listdir(scene_dir_path):
                paths.append(scene_dir_path + f"{os.sep}" + fi)

        return paths

    elif dataset_name == "trashcan": #COCO format
        #https://github.com/ultralytics/JSON2YOLO/blob/main/general_json2yolo.py
        images_path = data_path + f'{os.sep}val{os.sep}'
        paths = []
        for fi in os.listdir(images_path):
            paths.append(images_path + f"{os.sep}" + fi)

        return paths

    elif dataset_name == "lars": #mask
        images_path = data_path + f'{os.sep}images{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        return paths

    elif dataset_name == "rescuenet": #mask
        images_path = data_path + f'{os.sep}segmentation-validationset{os.sep}val-org-img{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        return paths

    elif dataset_name == "loveda": #mask
        images_path = data_path + f'{os.sep}Val{os.sep}Rural{os.sep}images_png{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        images_path = data_path + f'{os.sep}Val{os.sep}Urban{os.sep}images_png{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        return paths

    elif dataset_name == 'camvid':
        images_path = data_path + f'{os.sep}val{os.sep}'
        paths = []

        for fi in os.listdir(images_path):
            paths.append(images_path + fi)

        return paths


def process_mask(rgb_mask, colormap):
    output_mask = []

    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(rgb_mask, color), axis=-1)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)
    return output_mask

#https://gist.github.com/ilmonteux/8340df952722f3a1030a7d937e701b5a
def metrics_np(y_true, y_pred, metric_name, metric_type='standard', drop_last = True, mean_per_class=False, verbose=False):
    """ 
    Compute mean metrics of two segmentation masks, via numpy.
    
    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)
    
    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot 
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.
    
    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """
    
    assert y_true.shape == y_pred.shape, 'Input masks should be same shape, instead are {}, {}'.format(y_true.shape, y_pred.shape)
    assert len(y_pred.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(y_pred.shape)
    
    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')
    
    num_classes = y_pred.shape[-1]
    # if only 1 class, there is no background class and it should never be dropped
    drop_last = drop_last and num_classes>1
    
    if not flag_soft:
        if num_classes>1:
            # get one-hot encoded masks from y_pred (true masks should already be in correct format, do it anyway)
            y_pred = np.array([ np.argmax(y_pred, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)
            y_true = np.array([ np.argmax(y_true, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)
        else:
            y_pred = (y_pred > 0).astype(int)
            y_true = (y_true > 0).astype(int)
    
    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1,2) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) # or, np.logical_and(y_pred, y_true) #for one-hot # 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) # for one-hot # 
    
    if verbose:
        print('intersection (pred*true), intersection (pred&true), union (pred+true-inters), union (pred|true)')
        print(intersection, np.sum(np.logical_and(y_pred, y_true), axis=axes), union, np.sum(np.logical_or(y_pred, y_true), axis=axes))
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    
    metric = {'iou': iou, 'dice': dice}[metric_name]
    
    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask =  np.not_equal(union, 0).astype(int)
    # mask = 1 - np.equal(union, 0).astype(int) # True = 1
    
    if drop_last:
        metric = metric[:,:-1]
        mask = mask[:,:-1]
    
    # return mean metrics: remaining axes are (batch, classes)
    # if mean_per_class, average over batch axis only
    # if flag_naive_mean, average over absent classes too
    if mean_per_class:
        if flag_naive_mean:
            return np.mean(metric, axis=0)
        else:
            # mean only over non-absent classes in batch (still return 1 if class absent for whole batch)
            return (np.sum(metric * mask, axis=0) + smooth)/(np.sum(mask, axis=0) + smooth)
    else:
        if flag_naive_mean:
            return np.mean(metric)
        else:
            # mean only over non-absent classes
            class_count = np.sum(mask, axis=0)
            return np.mean(np.sum(metric * mask, axis=0)[class_count!=0]/(class_count[class_count!=0]))
    
def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = np.split(np.expand_dims(box1, 1), 2, 2), np.split(np.expand_dims(box2, 0), 2, 2)

    inter = (np.minimum(a2, b2) - np.maximum(a1, b1)).clip(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def mask_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [M, n] m2 means number of gt objects
    Note: n means image_w x image_h.

    return: masks iou, [N, M]
    """
    intersection = np.matmul(mask1.astype(np.int32), mask2.astype(np.int32).T).clip(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    # print(union)
    return intersection / (union + eps)

def process_batch(detections, labels, iouv):
    """
    Return a correct prediction matrix given detections and labels at various IoU thresholds.

    Args:
        detections (np.ndarray): Array of shape (N, 6) where each row corresponds to a detection with format
            [x1, y1, x2, y2, conf, class].
        labels (np.ndarray): Array of shape (M, 5) where each row corresponds to a ground truth label with format
            [class, x1, y1, x2, y2].
        iouv (np.ndarray): Array of IoU thresholds to evaluate at.

    Returns:
        correct (np.ndarray): A binary array of shape (N, len(iouv)) indicating whether each detection is a true positive
            for each IoU threshold. There are 10 IoU levels used in the evaluation.

    Example:
        ```python
        detections = np.array([[50, 50, 200, 200, 0.9, 1], [30, 30, 150, 150, 0.7, 0]])
        labels = np.array([[1, 50, 50, 200, 200]])
        iouv = np.linspace(0.5, 0.95, 10)
        correct = process_batch(detections, labels, iouv)
        ```

    Notes:
        - This function is used as part of the evaluation pipeline for object detection models.
        - IoU (Intersection over Union) is a common evaluation metric for object detection performance.
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return correct


def process_batch_mask(detections, labels, iouv, pred_masks=None, gt_masks=None):

    # if gt_masks.shape[1:] != pred_masks.shape[1:]:
    #     gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
    #     gt_masks = gt_masks.gt_(0.5)
    iou = mask_iou(gt_masks.reshape((gt_masks.shape[0], -1)), pred_masks.reshape((pred_masks.shape[0], -1)))

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return correct

def eval_detection_results(results, nc, input_size):

    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.size

    jdict, stats, ap, ap_class = [], [], [], []
    p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for pred, labels in results:
        if isinstance(pred, list):
            pred = np.array(pred)
        nl, npr = labels.shape[0], pred.shape[0]
        labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w=input_size, h=input_size)
        correct = np.zeros((npr, niou), dtype=np.bool)

        if npr == 0:
            if nl:
                stats.append((correct, *np.zeros((2, 0)), labels[:, 0]))
            
            continue
        
        # if single_cls:
        #         pred[:, 5] = 0

        if nl:
            # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            # labelsn = np.concatenate((labels[:, 0:1], tbox), 1)  # native-space labels
            # correct = process_batch(pred, labelsn, iouv)
            correct = process_batch(pred, labels, iouv)
        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))


    stats = [np.concatenate(x, 0) for x in zip(*stats)] 
    # print(stats[0].any())
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=".", names=())
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    #     # Print results
    s = ("%22s" + "%11s" * 6) % ("Class", "Instances", "P", "R", "mAP50", "mAP50-95", "F1")
    print(s)
    pf = "%22s" + "%11i" + "%11.3g" * 5  # print format
    # print(nt.sum(), mp, mr, map50, map, f1)
    if not isinstance(f1, float):
        f1 = f1.mean()
    print(pf % ("all", nt.sum(), mp, mr, map50, map, f1))
    return mp, mr, map50, map, f1

    # # Print results per class
    # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

def eval_mask_results(results, nc, input_size):

    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.size

    jdict, stats, ap, ap_class = [], [], [], []
    p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for pred_polygons, gt in results:

        labels, segments = gt
        nl, npr = labels.shape[0], len(pred_polygons)
        labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w=input_size, h=input_size)
        segments = [xyn2xy(segment,w=input_size, h=input_size) for segment in segments]
        gt_masks = polygons2masks((input_size, input_size), segments, color=1)
        correct = np.zeros((npr, niou), dtype=np.bool) 

        if npr == 0:
            if nl:
                stats.append((correct, *np.zeros((2, 0)), labels[:, 0]))
            
            continue

        #pred must be a array of shape (N, 6) where each row corresponds to a detection with format [x1, y1, x2, y2, conf, class]
        pred = []
        for polygon in pred_polygons:
            poly = np.array(polygon[2:]).reshape(-1, 2)
            label = polygon[0]
            score = polygon[1]
            x, y = poly.T
            pred.append([x.min(), y.min(), x.max(), y.max(), score, label])

        pred = np.array(pred)

        pred_masks = []
        for polygon in pred_polygons:
            pred_masks.append(np.array(polygon[2:]).reshape(-1, 2))

        pred_masks = polygons2masks((input_size, input_size), pred_masks, color=1)

        if nc == 1:
                pred[:, 5] = 0

        if nl:
            # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            # labelsn = np.concatenate((labels[:, 0:1], tbox), 1)  # native-space labels
            # correct = process_batch(pred, labelsn, iouv)
            correct = process_batch_mask(pred, labels, iouv, pred_masks, gt_masks)
            #print(correct)
        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))


    stats = [np.concatenate(x, 0) for x in zip(*stats)] 
    # print(stats)
    # print(stats[0].any())
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=".", names=())
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    #     # Print results
    s = ("%22s" + "%11s" * 6) % ("Class", "Instances", "P", "R", "mAP50", "mAP50-95", "F1")
    print(s)
    pf = "%22s" + "%11i" + "%11.3g" * 5  # print format
    # print(nt.sum(), mp, mr, map50, map, f1)
    if not isinstance(f1, float):
        f1 = f1.mean()
    print(pf % ("all", nt.sum(), mp, mr, map50, map, f1))
    return mp, mr, map50, map, f1

def convert_semantic_mask(mask, nc):
    if len(mask.shape) == 2 or (len(mask.shape) == 3 and mask.shape[-1] == 1):
        if nc == 3:
            mask = np.stack([mask==0, mask==1, mask==2], axis=-1).astype(np.float32)
        if nc == 8:
            mask = np.stack([mask==0, mask==1, mask==2, mask==3, mask==4, mask==5, mask==6, mask==7], axis=-1).astype(np.float32)
        if nc == 11:
            mask = np.stack([mask==0, mask==1, mask==2, mask==3, mask==4, mask==5, mask==6, mask==7, mask==8, mask==9, mask==10], axis=-1).astype(np.float32)

    mask = np.squeeze(mask)

    return mask

def eval_semantic_results(results, nc):

    y_pred = []
    y_true = []

    for pred, labels in results:
        pred = convert_semantic_mask(pred, nc)
        y_pred.append(pred)
        y_true.append(labels)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # if y_pred.shape != y_true.shape:
    #     y_pred = y_pred.transpose(0, 2, 3, 1)

    if y_pred.shape[3] != nc:
        print('Wrong class count!')
        return 0.0

    return metrics_np(y_true, y_pred, metric_name='dice')

def new_dice(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3)
    return dice

def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """Convert normalized segments into pixel segments, shape (n,2)."""
    y = np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y

def polygon2mask(img_size, polygons, color=1, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    mask = np.zeros(img_size, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (img_size[0] // downsample_ratio, img_size[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    # print(mask.shape)
    # if os.path.exists('./mask0.png'):
    #     cv2.imwrite('./mask1.png', mask)
    # else:
    #     cv2.imwrite('./mask0.png', mask)
    return mask

def polygons2masks(img_size, polygons, color, downsample_ratio=1):
    """
    Args:
        img_size (tuple): The image size.
        polygons (list[np.ndarray]): each polygon is [N, M],
            N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    masks = []
    for si in range(len(polygons)):
        mask = polygon2mask(img_size, [polygons[si].reshape(-1)], color, downsample_ratio)
        masks.append(mask)
    return np.array(masks)

def img2label_paths(img_paths, img_dir="images", lb_dir="labels", lb_ext=".txt"):
    """Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and
    extension with `.txt`.
    """
    sa, sb = f"{os.sep}{img_dir}{os.sep}", f"{os.sep}{lb_dir}{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + lb_ext for x in img_paths]   

# def img2label_paths_voc(img_paths):
#     """Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and
#     extension with `.txt`.
#     """
#     sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}annos{os.sep}"  # /images/, /labels/ substrings
#     return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".xml" for x in img_paths]   

# def img2label_paths_json(img_paths):
#     """Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and
#     extension with `.txt`.
#     """
#     sa, sb = f"{os.sep}val{os.sep}", f"{os.sep}annotations{os.sep}"  # /images/, /labels/ substrings
#     return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".json" for x in img_paths] 

def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def segments2boxes(segments):
    """Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)."""
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))                       # cls, xywh

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y 

def get_label(lb_file):
    with open(lb_file) as f:
        segments = []
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if any(len(x) > 6 for x in lb):  # is segment
            classes = np.array([x[0] for x in lb], dtype=np.float32)
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
            lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
        lb = np.array(lb, dtype=np.float32)
        if nl := len(lb):
            _, i = np.unique(lb, axis=0, return_index=True)
            if len(i) < nl:  # duplicate row check
                lb = lb[i]  # remove duplicates
                if segments:
                    segments = [segments[x] for x in i]

        return lb, segments

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def get_label_voc(lb_file, class_dict):
    #https://gist.github.com/Amir22010/a99f18ca19112bc7db0872a36a03a1ec
    tree = ET.parse(lb_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    lb = []

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if '_' in cls:
            parts = cls.split('_')
            cls = parts[0]
        cls_id = class_dict[cls]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        label = [cls_id] +  list(bb)
        lb.append(label)

    lb = np.array(lb, dtype=np.float32)
    return lb

def get_label_json(lb_file, class_dict):
    with open(lb_file) as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]

    shapes = data['shapes']

    lb = []
    segments = []

    for shape in shapes:
        if shape["shape_type"] == "polygon":
            cls = class_dict[shape["label"]]
            segment =  np.array(shape["points"], dtype=np.float32)/ np.array([w, h])
            label = np.concatenate((np.array([cls], dtype=np.float32), segments2boxes(segment)[0]))
            lb.append(label)
            segments.append(segment)

    lb = np.array(lb, dtype=np.float32)

    return lb, segments



# Plots ----------------------------------------------------------------------------------------------------------------


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    """Plots precision-recall curve, optionally per class, saving to `save_dir`; `px`, `py` are lists, `ap` is Nx2
    array, `names` optional.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    """Plots a metric-confidence curve for model predictions, supporting per-class visualization and smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
