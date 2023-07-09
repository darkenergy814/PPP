import os
import sys
import math
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

import torch
import cv2
import numpy as np


def yolov5Mask(ai_model, inputImg, outputImg, resultImg, showResult = True):
    model = torch.hub.load('ultralytics/yolov5', 'custom', ai_model)
    img = cv2.imread(inputImg)

    detections = model(img)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    results = detections.pandas().xyxy[0].to_dict(orient="records")

    for result in results:
        xmin = int(result['xmin'])
        ymin = int(result['ymin'])
        xmax = int(result['xmax'])
        ymax = int(result['ymax'])
        label = str(result['name'])
        mask[ymin:ymax, xmin:xmax] = 255
        img = annotator(img, xmin, ymin, xmax, ymax, label)
    if showResult:
        from google.colab.patches import cv2_imshow
        cv2_imshow(img)
    cv2.imwrite(resultImg, img)
    cv2_imshow(img)
    cv2.imwrite(outputImg, mask)
#yolov5Mask('yolov5s', 'maskTest.png', 'mask.png', 'bbox.png', True)

def mrcnnMask(inputImg, outputImg):
  # Root directory of the project
  ROOT_DIR = os.path.abspath("/content/PPP/Mask_RCNN") #절대경로 구하기

  # Import Mask RCNN
  sys.path.append(ROOT_DIR)  # To find local version of the library 파이썬 모듈경로 불러오기
  from mrcnn import utils
  import mrcnn.model as modellib
  from mrcnn import visualize
  # Import COCO config
  sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version 인수에 전달된 2개의 문자열을 결합하여, 1개의 경로로 할 수 있다
  import coco


  # Directory to save logs and trained model
  MODEL_DIR = os.path.join(ROOT_DIR, "logs")

  # Local path to trained weights file
  COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
  # Download COCO trained weights from Releases if needed
  if not os.path.exists(COCO_MODEL_PATH): #디렉토리나 파일 존재여부 확인
      utils.download_trained_weights(COCO_MODEL_PATH)

  # Directory of images to run detection on
  IMAGE_DIR = os.path.join(ROOT_DIR, "images")
  class InferenceConfig(coco.CocoConfig):
      # Set batch size to 1 since we'll be running inference on #각 iteration마다 주는 데이터 사이즈를 batch size
      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1

  config = InferenceConfig()
  config.display()
  # Create model object in inference mode.
  model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

  # Load weights trained on MS-COCO
  model.load_weights(COCO_MODEL_PATH, by_name=True)
  # Load a random image from the images folder
  image = skimage.io.imread(os.path.join("./", inputImg)) #이미지 불러오기
  print(image)

  # Run detection
  results = model.detect([image], verbose=0)  #보통 0 은 출력하지 않고, 1은 자세히, 2는 함축적인 정보만 출력
  #model파일에서 2482줄
  class_names = ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                                      '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                      '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                      '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                      '', '', '', '', '', '', '', '', '', '', '', '', '']
  r = results[0]
  index_list=list()
  for i, value in enumerate(results[0]['class_ids']):
    print(i,value)
    if value == 64:
      index_list.append(i)
    elif value == 63:
      index_list.append(i)
    elif value == 42:
      index_list.append(i)

  first = 0
  for i in index_list:
    maskImage = r['masks'][:,:,i].astype(np.uint8)
    if first == 0:
      first+=1
      maskResult = maskImage
      print(maskResult)
    else:
      maskResult = maskResult|maskImage
      print(maskResult)

  maskResult = maskResult * 255

  cv2.imwrite(outputImg, maskResult)

#utils --------------------------------------------------------------

def annotator(img, x1, y1, x2, y2, label):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    w, h = cv2.getTextSize(label, 1, fontScale=1, thickness=1)[0]
    outside = y1 - h >= 3
    p1 = x1, y1
    p2 = x1 + w, y1 - h - 1 if outside else y1 + h + 1
    cv2.rectangle(img, p1, p2, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(img,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                1,
                1,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA)
    return np.asarray(img)