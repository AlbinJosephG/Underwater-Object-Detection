from ultralytics import YOLO

DATA_YAML = "data.yaml"

def train():
    model = YOLO("yolov8n.pt")  # pretrained tiny model
    model.train(data=DATA_YAML, epochs=20, imgsz=320)
    print("Training complete! Weights saved.")

if __name__ == "__main__":
    train()
