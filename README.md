# CLIPpy
**Unofficial implementation of [Perceptual Grouping in Vision-Language Models](https://arxiv.org/abs/2210.09996)** paper.


## Demo
* [Colab Notebook](https://colab.research.google.com/drive/1iObo3WQHqHpEwUGmyHHQpRrOUqHoOZPF?usp=sharing) 
* [Gradio Demo](TBA)

## Setup
Refer to `requirements.txt`.


## Pre-trained Model
We replicate the work in CLIPpy paper under scaled down settings (due to resource and compute limitations). We use the CC-12M dataset (some files missing due to download issues) and use a single node with 8 V100 GPUs for training. Training for 5000 iterations, we are able to reach close to the reported segmentation performance on PASCAL VOC dataset. 
Our pre-trained checkpoint is provided [here](https://github.com/kahnchana/clippy/releases/download/v1.0/clippy_5k.pt).  

## Inference
We present examples of using CLIPpy in the following notebooks:
* [Basic inference](https://github.com/kahnchana/clippy/blob/master/notebooks/clippy.ipynb)
* [ImageNet & PASCAL VOC evaluation](https://github.com/kahnchana/clippy/blob/master/notebooks/evaluate.ipynb)
* WaterBirds evaluation (TBA)

## Analysis
We provide some analysis on the feature space of CLIPpy. 
* bird to different backgrounds vs average background (TBA)
* feature space visualization (TBA) 


## Training
We build over the [OpenCLIP](https://github.com/mlfoundations/open_clip) repository, use pre-trained checkpoint weights from [DINO](https://github.com/facebookresearch/dino) and [Sentence-T5](https://huggingface.co/sentence-transformers/sentence-t5-base), and follow the training schedule from the [CLIPpy paper](https://arxiv.org/abs/2210.09996). We also directly utilize the patch dropout implementation in OpenCLIP.

We run the train job as follows:
```bash
torchrun --nproc_per_node 8 -m training.main \
    --train-data "${ROOT}/data/cc12m/{00000..01242}.tar" \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 1024 \
    --precision fp32 \
    --grad-checkpointing \
    --workers 8 \
    --imagenet-val "${ROOT}/data/imagenet/val/" \
    --model "clippy-B-16" \
    --report-to "wandb" \
    --wandb-project-name "clip" \
    --zeroshot-frequency 1 \
    --name "clippy001"

```

## Citation

```
@article{clippy2022,
  title={Perceptual Grouping in Vision-Language Models},
  author={Kanchana Ranasinghe and Brandon McKinzie and Sachin Ravi and Yinfei Yang and Alexander Toshev and Jonathon Shlens},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.09996}
}
```
