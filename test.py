import os
import torch
import ultralytics
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Añadir esta línea

    device = 0 if torch.cuda.is_available() else "cpu"
    print(device)

    model_path: str = os.path.join(os.getcwd(), 'yolov8n.pt')

    model = ultralytics.YOLO(model_path)
    training = model.train(data='labeltool/YOLODataset/dataset.yaml', epochs=100, imgsz=640)
    # training = model.train(data='dataset.yaml', epochs=100, imgsz=640)





# results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)

# print(results[0].boxes.xyxy)

# print(results[0].boxes.conf)