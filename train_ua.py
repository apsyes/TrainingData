from ultralytics import YOLO

base_weights = "yolov8n.pt"  # or yolov8s.pt for more accuracy

data_yaml = r"C:\Users\you\db\Yolo\UA-DETRAC\data.yaml"

def main():
    model = YOLO(base_weights)

    model.train(
        data=data_yaml,
        epochs=30,        # keep it modest for first run
        imgsz=1280,
        batch=8,
        workers=2,        # ThinkPad-friendly
        patience=5,       # early stop if plateau
        cos_lr=True,
        plots=True,
    )

if __name__ == "__main__":
    main()
