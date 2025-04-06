# GAT

Pytorch implementation of a single GAT layer from the Paper: [Veličković et al., "Graph Attention Networks", 2018](https://arxiv.org/abs/1710.10903)

It uses low-level PyTorch tensor operations (without torch.nn) and includes a manual backpoperation through the GAT layer without PyTorch autograd.
The purpose of this repository is educational to understand how:
- message passing (Message, Aggregate, Update) works in (Attentional) Graph Neural Networks
- different Neural Network layers are built from basic tensor operations (MLP, (Inverted) Dropout, (Multi-head) Attention, Residual connections etc.)
- the gradients flow backward through the computation graph during optimization
