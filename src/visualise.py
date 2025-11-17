import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from dvclive.live import Live


def show_random_images(folder: str, n: int = 6, seed: int | None = None):
    """
    Display `n` random JPEG/PNG images from `folder` in a 2×3 grid.
    """

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_imgs = [p for p in Path(folder).iterdir()
                if p.suffix.lower() in img_exts]

    if len(all_imgs) < n:
        raise ValueError(
            f"Only {len(all_imgs)} image(s) found – need at least {n}.")

    rnd = random.Random(seed)
    chosen = rnd.sample(all_imgs, n)

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.ravel()                     # flatten for easy indexing

    for ax, img_path in zip(axes, chosen):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(img_path.name, fontsize=10)
        ax.axis('off')                       # no ticks

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    return plt.gcf()


def main():
    os.makedirs("img", exist_ok=True)
    train_img = show_random_images(os.path.join(
        "F1-Car-Recognition-1", "train", "images"), seed=42)
    val_img = show_random_images(os.path.join(
        "F1-Car-Recognition-1", "valid", "images"), seed=42)
    test_img = show_random_images(os.path.join(
        "F1-Car-Recognition-1", "test", "images"), seed=42)

    with Live("img") as live:
        live.log_image("img/train_sample.jpg", train_img)
        live.log_image("img/val_sample.jpg", val_img)
        live.log_image("img/test_sample.jpg", test_img)


if __name__ == "__main__":
    main()
