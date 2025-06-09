import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_model():
    try:
        model = YOLO("runs/detect/train/weights/best.pt")
    except Exception:
        model = YOLO("yolov8n.pt")
    return model

def draw_boxes_with_readable_text(img, boxes, class_names):
    height, width = img.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{class_names[cls]} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

        # Dynamically scale font size
        font_scale = max(min(width, height) / 1000, 0.8)
        font_thickness = 2

        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Draw text background
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)

        # Draw label
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return img

def detect(image, model):
    img_np = np.array(image.convert("RGB"))
    results = model(img_np)[0]
    img_np = draw_boxes_with_readable_text(img_np, results.boxes, model.names)
    return Image.fromarray(img_np)

def main():
    st.title("Underwater Object Detection")
    st.write("Upload an image and get detected objects with bounding boxes.")

    model = load_model()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect"):
            with st.spinner("Detecting..."):
                result_img = detect(image, model)
            st.image(result_img, caption="Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()
