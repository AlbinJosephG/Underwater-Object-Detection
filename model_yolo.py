from ultralytics import YOLO
import cv2
import os

# Paths
DATA_YAML = "data.yaml"  # Your dataset YAML file
MODEL_NAME = "yolov8n.pt"  # Pretrained YOLOv8 nano model (lightweight)
TRAINED_MODEL = "runs/detect/train/weights/best.pt"  # default best weights path after training

def train():
    model = YOLO(MODEL_NAME)
    # Train for 50 epochs, you can increase/decrease epochs
    model.train(data=DATA_YAML, epochs=50, imgsz=640)
    print("Training complete. Model saved at:", TRAINED_MODEL)

def load_model():
    # Load your best trained model or pretrained if training not done
    if os.path.exists(TRAINED_MODEL):
        print("Loading trained model:", TRAINED_MODEL)
        model = YOLO(TRAINED_MODEL)
    else:
        print("Loading pretrained model:", MODEL_NAME)
        model = YOLO(MODEL_NAME)
    return model

def predict_image(model, image_path):
    # Run prediction
    results = model(image_path)[0]

    # Load image for drawing
    img = cv2.imread(image_path)

    # Draw bounding boxes
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = model.names[cls]
        # Draw rectangle and put label text
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Save or return image with boxes
    output_path = "output.jpg"
    cv2.imwrite(output_path, img)
    return output_path

if __name__ == "__main__":
    # Uncomment this line if you want to train first
    # train()

    model = load_model()
    # Test with an example image
    img_path = "test/images/sample_underwater.jpg"  # change to your test image
    out_img = predict_image(model, img_path)
    print("Detection done. Output saved to", out_img)
