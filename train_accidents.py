from ultralytics import YOLO

data_dir = r"/nfs/stak/users/schaafa/trainingCapstone/data"

def main():
    # classification backbone
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data=data_dir,   # folder containing train/ and val/
        epochs=50,
        imgsz=224,
        batch=32,
        workers=30,
        patience=5,
    )

if __name__ == "__main__":
    main()
