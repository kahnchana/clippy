import torch
from open_clip import create_model_and_transforms, get_tokenizer

pretrained_ckpt = "/raid/kanchana/checkpoints/open_clip/clippy_best/clippy_5k.pt"
inet_path = "/raid/datasets/img1k_k/val"

tokenizor = get_tokenizer("clippy-B-16")
clippy, preprocess_train, preprocess_val = create_model_and_transforms(
    "clippy-B-16",
    precision='amp',
    device="cuda:0",
    pretrained=pretrained_ckpt
)

# sample_image = torch.rand(1, 3, 224, 224)
# image_embeddings = model.encode_image(sample_image)
#
# sample_text = ["cat", "dog", "bird", "tree", "plants", "green"]
# sample_text = [f"a photo of a {x}" for x in sample_text]
# text_vectors = tokenizor(sample_text)
# class_embeddings = model.encode_text(text_vectors)
#
# print(image_embeddings.shape, class_embeddings.shape)
#
# image_embeddings = model.encode_image(sample_image, pool=False)
# print(image_embeddings.shape, class_embeddings.shape)
#
# breakpoint()


from training.data import get_imagenet
from training.zero_shot import zero_shot_eval


class DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


args = DummyArgs(
    imagenet_val=inet_path,
    batch_size=16,
    workers=1,
    distributed=False,
    horovod=False,
    precision="amp",
    zeroshot_frequency=1,
    epochs=0,
    model="clippy-B-16",
    device="cuda:0"
)

# dataset = get_imagenet(args, (preprocess_train, preprocess_val), "val")
# data = {"imagenet-val": dataset}
#
# zero_shot_metrics = zero_shot_eval(clippy, data, 1, args)
# print(zero_shot_metrics)


from evaluation.datasets import PascalDataset
from evaluation.segmentation import PascalMIoU, PascalEvaluator


data_dir = "/raid/datasets/pascal_voc/VOC2012"
ds = PascalDataset(data_dir, transform=preprocess_val)
print(f"Loaded data")

dataloader = torch.utils.data.DataLoader(
    ds,
    batch_size=args.batch_size,
    num_workers=args.workers
)

metric = PascalMIoU()
evaluator = PascalEvaluator(model=clippy, dataset=ds, metric=metric, opts=args)
print(f"Setup evaluators")

evaluator.evaluate()
print(f"Finished evaluation")

res = evaluator.metric.compute()
print(res)

from timm.models.vision_transformer import VisionTransformer