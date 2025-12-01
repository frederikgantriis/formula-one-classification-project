import os

from dvc.api import params_show
from ultralytics import YOLO


def main():
    params = params_show()["demo"]

    model = YOLO(os.path.join("models", "best.pt"))

    model.track(params["url"], show=True)

if __name__ == "__main__":
    main()
