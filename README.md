# GAT

PyTorch implementation of a Graph Attentional layer from the paper: [Veličković et al., "Graph Attention Networks", 2018](https://arxiv.org/abs/1710.10903)

This implementation uses low-level PyTorch tensor operations (without `torch.nn`) and includes manual backpropagation through the GAT layer without PyTorch autograd.
This repository was created for educational purposes to understand how:
- message passing (Message, Aggregate, Update) works in (Attentional) Graph Neural Networks,
- different neural network layers are built from basic tensor operations (MLP, (inverted) dropout, (multi-head) attention, residual connections, etc.),
- gradients flow backward through the computation graph during optimization.
