from ultralytics import YOLO
import os
proxy = 'http://10.0.0.107:3128'
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

results = model.train(data='config.yaml', epochs = 10, imgsz=640)
