
# The idea
What if instead of complex sinusoidal or rotary position embeddings, I just used the binary representation of the coordinates? I know this works with complex-valued neural networks.


# Requirements
I develop inside of the January 2024 edition of the [Nvidia PyTorch Docker image](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01).
```docker run -it -d --gpus all -v /workspace:/workspace nvcr.io/nvidia/pytorch:24.01-py3```
