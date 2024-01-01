import math
from typing import Sequence, Optional, Union, Tuple

from PIL import Image
import requests
from io import BytesIO

import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import linen as nn

from transformers import FlaxT5EncoderModel, T5TokenizerFast, T5Config

from clippy_utils import get_similarity, vis_prediction, normalize_image
from layers import LayerNorm, Transformer

# # Set the JAX backend to CPU
jax.config.update('jax_platform_name', 'cpu')

class SentenceTransformer(nn.Module):
    out_features: int = 768
    bias: bool = False

    @nn.compact
    def __call__(self, input_ids, attention_mask):
        outputs = FlaxT5EncoderModel(config=T5Config.from_pretrained("t5-base"), name='transformer').module(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = jnp.mean(outputs.last_hidden_state, axis=1)
        pooled_output = nn.Dense(features=self.out_features, use_bias=self.bias, kernel_init=nn.initializers.xavier_uniform(), name='dense')(pooled_output)
        pooled_output = pooled_output / jnp.linalg.norm(pooled_output, axis=1, keepdims=True) 
        return pooled_output


class VisionTransformer(nn.Module):
  """Vision Transformer.

  Attributes:
    patch_size: The size of the patches to embed.
    features: Number of features.
    num_layers: Number of transformer blocks (self-attn + MLP).
    num_heads: Number of attention heads.
    out_features: Number of output features. If None, return transformer output.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading.
  """
  patch_size: int
  features: int
  num_layers: int
  num_heads: int
  out_features: Optional[int]
  use_underscore_module_name: bool = False

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    x = nn.Conv(self.features,
                kernel_size=(self.patch_size, self.patch_size),
                strides=(self.patch_size, self.patch_size),
                use_bias=True, name='conv1')(x)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    scale = 1.0 / jnp.sqrt(self.features)
    class_embedding = self.param('class_embedding',
                                 jax.nn.initializers.normal(stddev=scale),
                                 (self.features,))
    x = jnp.concatenate((jnp.tile(class_embedding[None, None, :],
                                  (x.shape[0], 1, 1)), x),
                        axis=1)
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.normal(stddev=scale),
                                      (x.shape[1], self.features))
    x = x + positional_embedding[None]

    # x = LayerNorm(name='ln_pre')(x)
    x = feature_map = Transformer(
        features=self.features,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        use_underscore_module_name=self.use_underscore_module_name,
        name='transformer')(
            x)

    if self.out_features is not None:
      x = LayerNorm(name='ln_post')(x[:, 0])
      x = nn.Dense(self.out_features, use_bias=False, name='proj')(x)
    else:
      x = LayerNorm(name='ln_post')(x)
    return x, feature_map  # pytype: disable=bad-return-type  # jax-ndarray


class CLIPpy(nn.Module):
  """Implements code for running inference with pre-trained CLIPpy model.

  Attributes:
    vocab_size: Size of the vocabulary.
    embed_dim: Size of the text and vision embeddings.
    text_features: Number of features in text transformer.
    text_num_layers: Number of text transformer blocks (self-attn + MLP).
    text_num_heads: Number of heads in text transformer.
    vision_features: Number of features in vision transformer.
    vision_num_layers: Number of vision transformer blocks (self-attn + MLP).
    vision_head_dim: Number of features per vision transformer attention head.
    vision_patch_size: Size of patches to embed in vision transformer.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading.
  """

  def setup(self):
      self.visual = VisionTransformer(
          patch_size=16,
          features=768,
          num_layers=12,
          num_heads=12,
          out_features=None)
      self.text = SentenceTransformer()
      self.logit_scale = self.param('logit_scale', jax.nn.initializers.constant(math.log(1 / 0.07)), ())

  def encode_image(self,
                   image: jnp.ndarray,
                   pool: str = 'all',
                   normalize: bool = True) -> jnp.ndarray:
    x, feature_map = self.visual(image)
    if pool == 'gmp':
        x = x[:, 1:].max(axis=-2)
    elif pool == 'cls':
        x = x[:, 0]
    else:  # pool == 'all'
        x = x[:, 1:] 
    if normalize:
      x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x

  def encode_text(self,
                  text: jnp.ndarray,
                  normalize: bool = True) -> jnp.ndarray:
    x = self.text(**text)
    return x

  def __call__(self,
               image: jnp.ndarray = None,
               text: list[str] = None,
               im_pool: str = 'all',
               normalize: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = y = None
    if image is not None:
      x = self.encode_image(image, pool=im_pool, normalize=normalize)
    if text is not None:
      y = self.encode_text(text, normalize)
    return x, y


if __name__ == "__main__":

  # Load and setup model
  model = CLIPpy()
  text_tokenizer = T5TokenizerFast.from_pretrained('t5-base')
  params = torch.load("/data/kanchana/models/clippy/flax/clippy_flax.pt")
#   params = jax.tree_util.tree_map(lambda p: p.cpu().numpy(), pretrained_ckpt)

  # Load image and text
  img_url = "https://epipoca.com.br/wp-content/uploads/2021/03/cf281f60a52896f2914116b85c74b809.jpg"
  response = requests.get(img_url)
  img = Image.open(BytesIO(response.content)).resize((224, 224))
  img_arr = np.array(img)
  img_arr = np.expand_dims(img_arr, axis=0)
  image = img_arr / 255.0
  image = normalize_image(image)

  sample_text = ["lantern", "girl", "background"]
  sample_prompts = [f"a photo of a {x}" for x in sample_text]
  text_tokenized = text_tokenizer(sample_prompts, max_length=77, truncation=True, return_tensors='np')

  # Run model
  im_embeddings, tx_embeddings = model.apply(params, text=text_tokenized, image=image)

  # Optional get only one embedding
  get_one = False
  if get_one:
    params = model.init(jax.random.PRNGKey(0), image=image)
    im_embeddings, _ = model.apply(params, image=image)

    params = model.init(jax.random.PRNGKey(0), text=text_tokenized)
    _, im_embeddings, tx_embeddings = model.apply(params, text=text_tokenized)

  # Visualize outputs
  similarity = get_similarity(torch.from_numpy(np.array(im_embeddings)), torch.from_numpy(np.array(tx_embeddings)), (224, 224), do_argmax=True)
  vis_prediction(sample_text, img_arr[0], similarity[0, 0])

