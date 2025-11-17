from roboflow import Roboflow
from roboflow_secrets import ROBOFLOWKEY


def main():
    rf = Roboflow(api_key=ROBOFLOWKEY)
    project = rf.workspace("main-803cc").project("f1-car-recognition-6b5om")
    version = project.version(1)
    dataset = version.download("yolov11")


if __name__ == "__main__":
    main()
