from dvc.api import params_show
from dvclive.live import Live
import os
from codecarbon import EmissionsTracker
from ultralytics import YOLO


def main():
    os.makedirs("models", exist_ok=True)

    params = params_show()["train"]

    tracker = EmissionsTracker()

    tracker.start()

    # TODO: Train model
    model = YOLO("yolo11n.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data=os.path.join(
        "F1-Car-Recognition-1", "data.yaml"), epochs=1)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model(os.path.join("F1-Car-Recognition-1", "valid",
                    "images", "00000009_jpg.rf.ae6b2b9a71e445a1533fbadf39387481.jpg"))

    # Export the model to ONNX format
    success = model.export(format="onnx")

    emissions = tracker.stop()

    with Live() as live:
        live.log_metric("CO2", emissions)


if __name__ == "__main__":
    main()
