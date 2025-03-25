import torch

class GAT():

    # Implementation of a Graph Attention Layer

    def __init__(self, in_dim, out_dim, alpha=0.2):

        # Glorot initialization from a uniform distribution
        limit_W = (6 / (in_dim + out_dim))**0.5
        self.W = (2*limit_W) * torch.rand(in_dim, out_dim) - limit_W
        self.b = torch.zeros(out_dim)

        limit_attention = (6 / (2 * out_dim + 1))**0.5
        self.attention = (2*limit_attention) * torch.rand(2*out_dim) -limit_attention
        self.alpha = alpha
        self.cache = ()

    def forward(self, H, A):

        N = H.shape[0]
        A_loop = A.clone()
        A_loop[torch.arange(N), torch.arange(N)] = 1

        h = H @ self.W + self.b

        hi = h.unsqueeze(1).expand(N, N, -1)
        hj = h.unsqueeze(0).expand(N, N, -1)

        concat = torch.cat((hi, hj), dim = -1)

        #e = (concat @ self.attention).squeeze(-1) # cant get backprop to be bit exact

        concat_att = self.attention * concat
        e = torch.sum(concat_att, dim = -1)

        eLeakyRelu = e.clone()
        eLeakyRelu[e < 0] *= self.alpha
        eInf = eLeakyRelu.clone()
        eInf[A_loop == 0] = float('-inf')

        e_max = torch.max(eInf, dim = 1, keepdim = True).values
        e_norm = eInf - e_max
        counts = torch.exp(e_norm)
        counts_sum = torch.sum(counts, dim = 1, keepdim = True)
        counts_sum_inv = counts_sum**-1
        att_normalized = counts * counts_sum_inv
        out = att_normalized @ h

        self.cache = (H, h, att_normalized, counts, counts_sum, counts_sum_inv, e_max, eInf, A_loop, e, self.alpha, concat_att, concat, self.attention)

        return out

    def backward(self, dout):
        (H, h, att_normalized, counts, counts_sum, counts_sum_inv, e_max, eInf, A_loop, e, alpha, concat_att, concat, attention) = self.cache

        N = H.shape[0]
        out_dim = h.shape[1]
        
        dh1 = att_normalized.T @ dout
        datt_normalized = dout @ h.T

        dexp_sum_inv = torch.sum(datt_normalized * counts, dim = 1, keepdim = True)
        dexp_sum = torch.sum(dexp_sum_inv * -(counts_sum**-2) , dim = 1, keepdim = True)

        dcounts1 = datt_normalized * counts_sum_inv
        dcounts2 = dexp_sum * torch.ones_like(counts)
        dcounts = dcounts1 + dcounts2

        de_norm = dcounts * counts

        dmax = torch.sum(de_norm, dim = 1, keepdim = True) * -torch.ones_like(e_max)

        deInf1 = de_norm.clone()
        deInf2 = torch.zeros_like(eInf)
        deInf2[torch.arange(N),torch.argmax(eInf, dim=1)] = 1
        deInf2 = deInf2 * dmax
        deInf = deInf1 + deInf2

        deLeakyRelu = deInf.clone()
        deLeakyRelu[A_loop == 0] = 0

        de = deLeakyRelu.clone()
        de[e < 0] *= alpha

        # backprop if e = (concat @ self.attention).squeeze(-1) was used:
        # de_unsqueeze = de.unsqueeze(-1)
        # dattention = torch.sum(concat * de_unsqueeze, dim = (0, 1))
        # dconcat = de_unsqueeze * attention
        
        dconcat_att = torch.broadcast_to(torch.unsqueeze(de, -1), concat_att.shape)
        dattention = torch.sum(concat * dconcat_att, dim = (0, 1))
        dconcat = dconcat_att * attention

        dhi = dconcat[:, :, :out_dim]
        dhj = dconcat[:, :, out_dim:]

        dh2 = torch.sum(dhi.view(N, N, out_dim), dim=1)
        dh3 = torch.sum(dhj.view(N, N, out_dim), dim=0)
        dh = dh3 + dh2 + dh1 # numerically inaccurate since adding large and small magnitudes together

        dW = H.T @ dh
        db = torch.sum(dh, dim = 0)
        dH = dh @ self.W.T

        return dH, dW, db, dattention