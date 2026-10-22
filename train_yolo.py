from ultralytics import YOLO

def main():
    # Load lightweight pre-trained model
    model = YOLO("yolov8n.pt")

    # Train on your Roboflow dataset
    model.train(
        data="loco_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # Uses your NVIDIA GPU
        name="loco_plate_detector"
    )

if __name__ == "__main__":
    main()