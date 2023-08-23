from ultralytics import YOLO
import yaml
import sys
sys.path.append('../../')
import logging.config

# Load a model
model = YOLO('D:/hh/detect/src/runs/detect/yolov8n_custom/weights/test.pt')  # load a partially trained model,copy best weight.pt to test.pt

# Resume training
model.train(resume=True)