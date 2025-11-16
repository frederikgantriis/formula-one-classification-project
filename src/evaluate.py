import os
from dvc.api import params_show
from dvclive.live import Live


def main():
    params = params_show()["evaluate"]

    # TODO: Load model

    # TODO: Load test dataset

    # TODO: Calculate metrics based on predicted vs truth

    with Live() as live:
        pass


if __name__ == "__main__":
    main()
