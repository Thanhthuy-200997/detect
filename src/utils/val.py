from ultralytics import YOLO

def val(model_val,data,imgsz,batch):
   print('----------Get start validate model------------')
   # Load the model.
   # model = YOLO('yolov8n.pt')
   model = YOLO(model_val)
   
   # Training.
   results = model.val(
      data=data,
      imgsz=imgsz,
      batch=batch
   )
   print('Training validate complete.Please check in val folder!!!')
   return results