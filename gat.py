import torch

class GAT():

     # Implementation of a Graph Attentional layer based on the Graph Attention Network paper (Veličković et al., 2018).

    def __init__(self, in_dim, out_dim, num_heads, alpha=0.2, in_drop=0.6, coef_drop=0.6, concat = True, skip = True, generator=None):

        # Glorot initialization from a uniform distribution
        limit_W = (6 / (in_dim + out_dim))**0.5
        self.W = (2*limit_W) * torch.rand(in_dim, out_dim * num_heads, generator=generator) - limit_W
        self.b = torch.zeros(out_dim * num_heads)

        limit_attention = (6 / (2 * out_dim + 1))**0.5
        self.attention = (2*limit_attention) * torch.rand(num_heads, 2*out_dim, generator=generator) - limit_attention
        self.num_heads = num_heads

        self.in_drop = in_drop
        self.coef_drop = coef_drop
        self.alpha = alpha
        self.concat = concat
        self.skip = skip
        self.skip_proj = None

        # Adding projection for skip connection if in_dim != out_dim
        if self.skip and in_dim != out_dim:
            limit_skip_proj = (6 / (in_dim + out_dim))**0.5
            self.skip_proj = (2*limit_skip_proj) * torch.rand(in_dim, out_dim, generator=generator) - limit_skip_proj

        self.cache = ()

        self.generator=generator

    def forward(self, H, A, mode_train = True): 

        # Forward pass: concatenation or averaging of aggregated features from all heads

        N = H.shape[0]
        A_loop = A.clone()
        A_loop[torch.arange(N), torch.arange(N)] = 1

        # Dropout on input
        H_dropout_mask = None
        if mode_train:
            H_dropout_mask = (torch.rand(H.shape, generator=self.generator) < (1.0-self.in_drop)) / (1.0-self.in_drop)
            H_dropped = H * H_dropout_mask
        else:
            H_dropped = H

        h = H_dropped @ self.W

        # hi ‖ hj for all node pairs (i, j)
        h_reshaped = h.view(N, self.num_heads, -1)
        hi = h_reshaped.unsqueeze(1).expand(N, N, self.num_heads, -1)
        hj = h_reshaped.unsqueeze(0).expand(N, N, self.num_heads, -1)
        concat = torch.cat((hi, hj), dim = -1)

        # Performing self-attention
        e = torch.einsum('ijhf, hf ->ijh', concat, self.attention)
        eLeakyRelu = e.clone()
        eLeakyRelu[e < 0] *= self.alpha

        # Masking for non-adjacent nodes
        eInf = eLeakyRelu.clone()
        eInf[A_loop == 0] = float('-inf') 

        # Attention normalization via softmax
        e_max = torch.max(eInf, dim = 1, keepdim = True).values
        e_norm = eInf - e_max
        counts = torch.exp(e_norm)
        counts_sum = torch.sum(counts, dim = 1, keepdim = True)
        counts_sum_inv = counts_sum**-1
        att_normalized = counts * counts_sum_inv

        # Dropout on normalized attention coefficients
        att_normalized_dropped = None
        att_dropout_mask = None
        if mode_train:
            att_dropout_mask = (torch.rand(att_normalized.shape, generator=self.generator) < (1.0-self.coef_drop)) / (1.0-self.coef_drop)
            att_normalized_dropped = att_normalized * att_dropout_mask 
        else:
            att_normalized_dropped = att_normalized

        # Feature aggregation
        g = torch.einsum('njh, jhf -> nhf', att_normalized_dropped, h_reshaped)

        # Residual connection with linear transformation if in_dim != out_dim
        g_skip = g.clone()
        if self.skip:
            if self.skip_proj is None:
                H_expand = H.unsqueeze(1).expand(N, self.num_heads, -1)
                g_skip += H_expand
            else:
                H_skip_proj = H @ self.skip_proj
                H_expand = H_skip_proj.unsqueeze(1).expand(N, self.num_heads, -1)
                g_skip += H_expand

        # Adding bias
        g_skip_bias = g_skip + self.b.view(self.num_heads, -1)

        # Output: concatenate or average from each head
        out = None
        if self.concat:
            out = g_skip_bias.reshape(N, -1)
        else:
            out = g_skip_bias.mean(dim = 1)

        self.cache = (self.W, g_skip_bias, H, h_reshaped, att_normalized_dropped, mode_train, att_dropout_mask, counts, counts_sum, counts_sum_inv, e_max, eInf, A_loop, concat, e, h, H_dropped, H_dropout_mask)

        return out

    
    def backward(self, dout):

        # Backward pass to compute the gradient w.r.t. H, W, b, attention, skip_proj
        
        (W, g_skip_bias, H, h_reshaped, att_normalized_dropped, mode_train, att_dropout_mask, counts, counts_sum, counts_sum_inv, e_max, eInf, A_loop, concat, e, h, H_dropped, H_dropout_mask) = self.cache

        out_dim = W.shape[1] // self.num_heads

        # Gradient through concatenation/averaging
        dg_skip_bias = None
        if self.concat:
            dg_skip_bias = dout.reshape(g_skip_bias.shape)
        else:
            dg_skip_bias = (dout.unsqueeze(1) * (self.num_heads**-1)) * torch.ones(g_skip_bias.shape)

        db = dg_skip_bias.sum(dim=0).view(self.b.shape)
        dg_skip = dg_skip_bias.clone()

        # Gradient w.r.t. residual connection
        dskip_proj = None
        dH1 = None
        if self.skip:
            if self.skip_proj is None:
                dH_expand = dg_skip.clone()
                dH1 = dH_expand.sum(dim=1)
            else:
                dH_expand = dg_skip.clone()
                dH_skip_proj = torch.sum(dH_expand, dim=1)
                dskip_proj = H.T @ dH_skip_proj
                dH1 = dH_skip_proj @ self.skip_proj.T

        # Gradient through feature aggregation
        dg = dg_skip.clone()
        datt_normalized_dropped = torch.einsum('nhf,jhf->njh', dg, h_reshaped)
        dh_reshaped1 = torch.einsum('nhf,njh->jhf', dg, att_normalized_dropped)

        # Gradient through attention layer
        datt_normalized = None
        if mode_train:
            datt_normalized = datt_normalized_dropped * att_dropout_mask
        else:
            datt_normalized = datt_normalized_dropped

        dcounts_sum_inv = torch.sum(datt_normalized * counts, dim = 1, keepdim = True)
        dcounts_sum = torch.sum(dcounts_sum_inv * -(counts_sum**-2) , dim = 1, keepdim = True)
        dcounts1 = counts_sum_inv * datt_normalized
        dcounts2 = torch.ones_like(counts) * dcounts_sum
        dcounts = dcounts2 + dcounts1
        de_norm = dcounts * counts
        de_max = torch.sum(de_norm, dim = 1, keepdim = True) * -torch.ones_like(e_max)

        deInf1 = de_norm.clone()
        deInf2 = torch.zeros_like(eInf)
        deInf2.scatter_(1, torch.argmax(eInf, dim=1, keepdim=True), 1.0)
        deInf2 = deInf2 * de_max

        deInf = deInf1 + deInf2
        deLeakyRelu = deInf.clone()
        deLeakyRelu[A_loop == 0] = 0

        de = deLeakyRelu.clone()
        de[e < 0] *= self.alpha
        dattention = torch.einsum('ijh, ijhf -> hf', de, concat)
        
        # Grad w.r.t. concat(hi, hj)
        dconcat = de.unsqueeze(-1) * self.attention
        dhi = dconcat[:,:,:,:out_dim]
        dhj = dconcat[:,:,:,out_dim:]

        # Grad w.r.t. input features H
        dh_reshaped2 = torch.sum(dhi, dim=1)
        dh_reshaped3 = torch.sum(dhj, dim=0)
        dh_reshaped = dh_reshaped3 + dh_reshaped1 + dh_reshaped2 # Could be numerically inaccurate
        dh = dh_reshaped.view(h.shape)
        dH_dropped = dh @ self.W.T
        dW = H_dropped.T @ dh
        dH2 = None

        if mode_train:
            dH2 = dH_dropped * H_dropout_mask
        else:
            dH2 = dH_dropped

        dH = dH2
        if dH1 is not None:
            dH += dH1

        return dH, dW, db, dattention, dskip_proj