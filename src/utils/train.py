from ultralytics import YOLO


def train(model,data,imgsz,epochs,batch,save_model):
   print('--------- Get start training model -------------')
   # Load the model.
   model = YOLO(model)
   
   # Training.
   results = model.train(
      data=data,
      imgsz=imgsz,
      epochs=epochs,
      batch=batch,
      name=save_model
   )
   print('Training model complete.Please check in train folder!!!')
   return results