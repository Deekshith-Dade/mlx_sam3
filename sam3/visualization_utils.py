import mlx.core as mx
import numpy as np

from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

def generate_colors(n_colors=256, n_samples=5000):
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    # print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    # print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


COLORS = generate_colors(n_colors=128, n_samples=5000)

def draw_box_on_image(image, box, color=(0, 255, 0)):
    """
    Draws a rectangle on a given PIL image using the provided box coordinates in xywh format.
    :param image: PIL.Image - The image on which to draw the rectangle.
    :param box: tuple - A tuple (x, y, w, h) representing the top-left corner, width, and height of the rectangle.
    :param color: tuple - A tuple (R, G, B) representing the color of the rectangle. Default is red.
    :return: PIL.Image - The image with the rectangle drawn on it.
    """
    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    # Unpack the box coordinates
    x, y, w, h = box
    x, y, w, h = int(x), int(y), int(w), int(h)
    # Get the pixel data
    pixels = image.load()
    # Draw the top and bottom edges
    for i in range(x, x + w):
        pixels[i, y] = color
        pixels[i, y + h - 1] = color
        pixels[i, y + 1] = color
        pixels[i, y + h] = color
        pixels[i, y - 1] = color
        pixels[i, y + h - 2] = color
    # Draw the left and right edges
    for j in range(y, y + h):
        pixels[x, j] = color
        pixels[x + 1, j] = color
        pixels[x - 1, j] = color
        pixels[x + w - 1, j] = color
        pixels[x + w, j] = color
        pixels[x + w - 2, j] = color
    return image


def plot_bbox(
    img_height,
    img_width,
    box,
    box_format="XYXY",
    relative_coords=True,
    color="r",
    linestyle="solid",
    text=None,
    ax=None,
):
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        facecolor = "w"
        ax.text(
            x,
            y - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},
        )

def plot_mask(mask, color="r", ax=None):
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)

def normalize_bbox(bbox_xywh, img_w, img_h):
    # Assumes bbox_xywh is in XYWH format
    if isinstance(bbox_xywh, list):
        assert (
            len(bbox_xywh) == 4
        ), "bbox_xywh list must have 4 elements. Batching not support except for torch tensors."
        normalized_bbox = bbox_xywh
        normalized_bbox[0] /= img_w
        normalized_bbox[1] /= img_h
        normalized_bbox[2] /= img_w
        normalized_bbox[3] /= img_h
    else:
        assert isinstance(
            bbox_xywh, mx.array
        ), "Only torch tensors are supported for batching."
        normalized_bbox = bbox_xywh
        assert (
            normalized_bbox.shape[-1] == 4
        ), "bbox_xywh tensor must have last dimension of size 4."
        normalized_bbox[..., 0] /= img_w
        normalized_bbox[..., 1] /= img_h
        normalized_bbox[..., 2] /= img_w
        normalized_bbox[..., 3] /= img_h
    return normalized_bbox

def plot_results(img, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    # Convert MLX tensors to NumPy so Matplotlib gets plain Python scalars
    boxes = np.asarray(results["boxes"])
    masks = np.asarray(results["masks"])
    scores = np.asarray(results["scores"])
    nb_objects = len(scores)
    print(f"found {nb_objects} object(s)")
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(masks[i].squeeze(0), color=color)
        w, h = img.size
        prob = float(scores[i].item())
        plot_bbox(
            h,
            w,
            boxes[i],
            text=f"(id={i}, {prob=:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )