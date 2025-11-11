from ultralytics import YOLO

model = YOLO('yolov8m.yaml')

def training_model():
    print("Training.....")
    results = model.train(
        data= 'D:/license_plate_detection/data/detection_plate.yaml',
        epochs= 150,
        imgsz= 640,
        batch= 8,
        device=0,
        mosaic= 1.0,
        mixup=0.2,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        degrees=10,
        shear=2,
        translate=0.1,
        name='lpd_yolo',
        patience= 50,
        optimizer='AdamW',
        lr0=0.01
    )
    print("Completed training process!")
    print("Your weights had been saved in: /lpd_yolo/weights/best.pt")

if __name__=='__main__':
    training_model()