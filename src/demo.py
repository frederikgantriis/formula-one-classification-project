import os

from dvc.api import params_show
from ultralytics import YOLO


def main():
    params = params_show()["demo"]

    model = YOLO(os.path.join("models", "best.pt"))

    model.track("https://youtu.be/uQc-pW3QLuI?si=-HpDI2v80x5_Rt7c", show=True)

if __name__ == "__main__":
    main()
