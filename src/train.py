from dvc.api import params_show
from dvclive.live import Live
import os
from codecarbon import EmissionsTracker


def main():
    os.makedirs("models", exist_ok=True)

    params = params_show()["train"]

    tracker = EmissionsTracker()

    tracker.start()

    # TODO: Train model

    emissions = tracker.stop()

    with Live() as live:
        live.log_metric("CO2", emissions)


if __name__ == "__main__":
    main()
