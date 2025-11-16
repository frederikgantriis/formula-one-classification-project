import os
from dvc.api import params_show
from dvclive.live import Live


def main():

    # Create path for dvc output
    os.makedirs("transformed_data", exist_ok=True)

    # Fetch parameters
    params = params_show()["transform"]

    # TODO: Load dataset

    # TODO: Process

    # TODO: Save all datasets


if __name__ == "__main__":
    main()
