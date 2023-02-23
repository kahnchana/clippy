import os
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from .datasets import PascalDataset


class PascalEvaluator:

    def __init__(self):
        self._num_classes = 20 + 1  # background
        self.confusion_matrix = None
        self._num_examples = 0

        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self._num_classes,) * 2, dtype=np.int64)
        self._num_examples = 0

    def update(self, prediction, label):
        self._num_examples += label.shape[0]
        mask = label < 255  # skip-pixel for VOC dataset
        indices = self._num_classes * label[mask] + prediction[mask]
        m = np.bincount(indices, minlength=self._num_classes ** 2).reshape(
            self._num_classes, self._num_classes
        )
        self.confusion_matrix += m

    def compute(self):
        if self._num_examples == 0:
            return 0

        cm = self.confusion_matrix
        iou_list = cm.diagonal() / (
                cm.sum(axis=0) + cm.sum(axis=1) - cm.diagonal() + np.finfo(np.float32).eps
        )
        return {"mIoU": np.nanmean(iou_list), "per_class": iou_list.tolist()}


def load_pred_from_npy(path, resize):
    pred = np.load(path)
    pred_resized = [np.ones(resize) * 0.5, ]
    for i in pred:
        pred_resized.append(np.array(Image.fromarray(i).resize(tuple(reversed(resize)))))
    pred_resized = np.stack(pred_resized, axis=0)
    pred_resized = np.argmax(pred_resized, axis=0)
    for idx, real_label in enumerate(sample[2]):
        pred_resized[pred_resized == (idx + 1)] = real_label
    return pred_resized


def save_vis(image, label, prediction, save):
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Image", fontdict={'fontsize': 28})
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(image)
    plt.imshow(prediction, alpha=0.5)
    plt.axis("off")
    plt.title("Prediction", fontdict={'fontsize': 28})
    plt.subplot(133)
    vis_gt = label.copy()
    vis_gt[vis_gt == 255] = 0
    plt.imshow(vis_gt, cmap="tab20b", interpolation="nearest")
    plt.axis("off")
    plt.title("GT Label", fontdict={'fontsize': 28})
    plt.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':

    pred_list = glob.glob(f"/Users/kanchana/Downloads/results/*.npy")
    data_dir = "/Users/kanchana/Downloads/VOCdevkit/VOC2012"
    ds = PascalDataset(data_dir)
    evaluator = PascalEvaluator()
    save_path = ""

    for pred_path in pred_list:
        name = os.path.basename(pred_path).split(".")[0]
        sample = ds[name]
        new_size = sample[1].shape

        pred = load_pred_from_npy(pred_path, new_size)
        gt = sample[1]

        save_vis(sample[0], sample[1], pred, f"{save_path}/{name}.png" if save_path else save_path)

        evaluator.update(prediction=pred.astype(np.int32), label=gt.astype(np.int32))

    res = evaluator.compute()
