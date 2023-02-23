import torch
from open_clip import create_model_and_transforms, get_tokenizer

pretrained_ckpt = "/raid/kanchana/checkpoints/open_clip/clippy002/checkpoints/epoch_10.pt"
# pretrained_ckpt = "/raid/kanchana/checkpoints/open_clip/clippy_longer/pretrained_clippy_5k.pt"

tokenizor = get_tokenizer("clippy-B-16")
model, preprocess_train, preprocess_val = create_model_and_transforms(
        "clippy-B-16",
    )
        # pretrained=pretrained_ckpt

ckpt = torch.load(pretrained_ckpt)
ckpt = {k[7:]: v for k, v in ckpt['state_dict'].items()}
msg = model.load_state_dict(ckpt, strict=False)

sample_image = torch.rand(1, 3, 224, 224)
image_embeddings = model.encode_image(sample_image)

sample_text = ["cat", "dog", "bird", "tree", "plants", "green"]
sample_text = [f"a photo of a {x}" for x in sample_text]
text_vectors = tokenizor(sample_text)
class_embeddings = model.encode_text(text_vectors)

print(image_embeddings.shape, class_embeddings.shape)

image_embeddings = model.encode_image(sample_image, pool=False)
print(image_embeddings.shape, class_embeddings.shape)

breakpoint()
