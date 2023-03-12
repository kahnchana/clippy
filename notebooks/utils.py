import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt


def get_similarity(image_encodings, label_encodings, target_shape, interpolation="bilinear", do_argmax=False):
    """

    Args:
        image_encodings:
        label_encodings:
        target_shape:
        interpolation: nearest, bilinear
        do_argmax:

    Returns:

    """

    image_encodings = image_encodings.cpu()
    label_encodings = label_encodings.cpu()

    image_encodings = rearrange(
        image_encodings, "b (h w) d -> d b h w", h=int(np.sqrt(image_encodings.shape[-2]))
    )
    # assuming square inputs & targets
    scale_ratio = (target_shape[-2] / image_encodings.shape[-2],
                   target_shape[-1] / image_encodings.shape[-1],)
    temp_list = []
    for i in image_encodings:
        i = i.unsqueeze(1)
        i = torch.nn.functional.interpolate(
            i, scale_factor=scale_ratio, mode=interpolation
        )
        temp_list.append(i)
    image_encodings = torch.cat(temp_list, dim=1)

    image_encodings = rearrange(image_encodings, "b d h w -> b h w d")
    similarity = image_encodings @ label_encodings.T
    similarity = rearrange(similarity, "b h w d-> b d h w")
    if do_argmax:
        similarity = torch.argmax(similarity, dim=1, keepdim=True).to(torch.float64)
    return similarity


def get_cmap(ncolors):
    if ncolors > 9:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.tab10
    cmaplist = [cmap(i) for i in range(ncolors)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", cmaplist, ncolors)

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)

    return cmap, mappable


def vis_prediction(sample_text, img_arr, similarity):
    N = len(sample_text)
    cmap, mappable = get_cmap(N)

    fig, axs = plt.subplots(1, 2)

    _ = axs[0].imshow(img_arr)
    _ = axs[1].imshow(img_arr)
    _ = axs[1].imshow(similarity, cmap=cmap, interpolation="nearest", vmin=0, vmax=N, alpha=0.5)
    axs[0].axis("off")
    axs[1].axis("off")

    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.0, 0.85, 1.0, 0.05])
    colorbar = plt.colorbar(mappable, cax=cbar_ax, cmap=cmap, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0, N, N))
    colorbar.set_ticklabels(sample_text)


class DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
