import torch

class GCN():

    # Implementation of a Graph Convolutional Layer

    def __init__(self, in_dim, out_dim, alpha=0.0, generator=None):


        # self.W1 for neighborhood aggregation
        # self.W2 for transforming hidden vector of self

        # Kaiming initialization from a normal distribution
        self.W1 = torch.randn(in_dim, out_dim, generator=generator) * (2.0 / (in_dim * (1 + alpha**2)))**0.5
        self.W2 = torch.randn(in_dim, out_dim, generator=generator) * (2.0 / (in_dim * (1 + alpha**2)))**0.5
        self.b = torch.zeros(out_dim)
        self.alpha = alpha
        self.cache = ()

    def forward(self, H, A):

        A_norm = self._getAnorm(A)
        h1 = H @ self.W1
        h11 = A_norm @ h1
        h2 = H @ self.W2
        h = h11 + h2 + self.b
        leakyRelu = h.clone()
        leakyRelu[h < 0] *= self.alpha

        self.cache = (h, self.alpha, H, self.W2, A_norm, self.W1)

        return leakyRelu

    def backward(self, dout):
        (h, alpha, H, W2, A_norm, W1) = self.cache
        dh = dout.clone()
        dh[h < 0] *= alpha

        dh11 = dh.clone()
        dh2 = dh.clone()
        db = torch.sum(dh, dim = 0)

        dH1 = dh2 @ W2.T
        dW2 = H.T @ dh2

        dh1 = A_norm.T @ dh11

        dW1 = H.T @ dh1
        dH2 = dh1 @ W1.T

        dH = dH1 + dH2

        return dH, dW1, dW2, db
    
    def _getAnorm(self, A):

        # Symmetric normalization of the adjacency matrix

        N = A.shape[0]

        D_inv_sqrt = torch.zeros_like(A)
        A_sum = torch.sum(A, dim = 1)
        A_sum_inv_sqrt = A_sum**-0.5
        D_inv_sqrt[torch.arange(N), torch.arange(N)] = A_sum_inv_sqrt
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return A_norm