from ultralytics import YOLO     
model = YOLO('ts2.pt')     
results = model.export(format='engine') 
