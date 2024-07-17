import os
import pandas as pd
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import yaml
import torch
from tqdm.auto import tqdm
import shutil

import ultralytics
ultralytics.checks()
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model

img = 'data/images/val/coronoid-process-fracture_jpg.rf.71650459c69a9734ecd545067cf18bf4.jpg'

from PIL import Image

image_path = img
image = Image.open(image_path)

image_array = np.array(image)

image_dtype = image_array.dtype

print("Image data type:", image_dtype)

plt.imshow(image_array)

results = model.predict(image_array, conf = 0.5)

for result in results:
    boxes = result.boxes

boxes

class_names = model.names


fig, ax = plt.subplots()


ax.imshow(image_array)


for box, conf, class_idx in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):

    class_index = int(class_idx)


    class_name = class_names[class_index]


    x1, y1, x2, y2 = box[:4]


    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')


    ax.add_patch(rect)


    label = f"{class_name}: {conf:.2f}"
    ax.text(x1, y1 - 10, label, fontsize=15, color='b')


plt.show()

train_data ="training"
csv_data = "train_solution_bounding_boxes (1).csv"
test_data ="testing"

if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists("data/images"):
    os.mkdir("data/images")
if not os.path.exists("data/images/train"):
    os.mkdir("data/images/train")
if not os.path.exists("data/images/val"):
    os.mkdir("data/images/val")
if not os.path.exists("data/labels"):
    os.mkdir("data/labels")
if not os.path.exists("data/labels/train"):
    os.mkdir("data/labels/train")
if not os.path.exists("data/labels/val"):
    os.mkdir("data/labels/val")

root_dir="data"
labels_dir="data/labels"
images_dir="data/images"

df=pd.read_csv(csv_data)

width=676
height=380

df["class"]=0
df.rename(columns={'image':'img_name'}, inplace=True)

df["x_centre"]=(df["xmin"]+df["xmax"])/2
df["y_centre"]=(df["ymin"]+df["ymax"])/2
df["width"]=(df["xmax"]-df["xmin"])
df["height"]=(df["ymax"]-df["ymin"])


df["x_centre"]=df["x_centre"]/width
df["y_centre"]=df["y_centre"]/height
df["width"]=df["width"]/width
df["height"]=df["height"]/height

df_yolo=df[["img_name","class","x_centre","y_centre","width","height"]]
df_yolo.sample(5)

img_list = list(sorted(os.listdir(train_data)))
np.random.shuffle(img_list)

for i, img_name in enumerate(img_list):
    subset = "train"
    if i >= 80 / 100 * len(img_list):
        subset = "val"

    if np.isin(img_name, df_yolo['img_name']):
        columns = ["class", "x_centre", "y_centre", "width", "height"]
        img_box = df[df['img_name'] == img_name][columns].values
        label_path = os.path.join(labels_dir, subset, img_name[:-4] + ".txt")
        with open(label_path, "w+") as f:
            for row in img_box:
                text = " ".join(row.astype(str))
                f.write(text)
                f.write("\n")

    old_image_path = os.path.join(train_data, img_name)
    new_image_path = os.path.join(images_dir, subset, img_name)
    shutil.copy(old_image_path, new_image_path)

    a = 'data/images/train'
    b = 'data/labels/train'

    if not 'files_counted' in globals():
        num_files_a = len(os.listdir(a))
        num_files_b = len(os.listdir(b))

        print(f'number of files of images in train folder: {num_files_a}')
        print(f'number of files of labels in train folder: {num_files_b}')

    globals()['files_counted'] = True

    yolo_format = dict(path="",
                       train="C:/Users/Baran/Desktop/yolov8/yolov8/data/images/train",
                       val="C:/Users/Baran/Desktop/yolov8/yolov8/data/images/val",
                       nc=7,
                       names={0: "elbow positive", 1:"fingers positive", 2:"forearm fracture", 3:"humerus fracture", 4:"humerus", 5:"shoulder fracture", 6:"wrist positive",})

    with open('data.yaml', 'w') as outfile:
        yaml.dump(yolo_format, outfile, default_flow_style=False)

model.train(data="data.yaml", epochs=25, patience=5, batch=8,
            lr0=0.0005, imgsz=640)
