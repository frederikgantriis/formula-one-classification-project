from dvc.api import params_show
from dvclive.live import Live
import os
from codecarbon import EmissionsTracker
from ultralytics import YOLO


def main():
    os.makedirs("models", exist_ok=True)

    params = params_show()["train"]

    model_name = params["model_name"]
    epochs = params["epochs"]

    tracker = EmissionsTracker()

    tracker.start()

    model = YOLO(model_name)

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    model.train(data=os.path.join(
        "F1-Car-Recognition-1", "data.yaml"), epochs=epochs)

    emissions = tracker.stop()

    with Live() as live:
        live.log_metric("CO2", emissions)


if __name__ == "__main__":
    main()
