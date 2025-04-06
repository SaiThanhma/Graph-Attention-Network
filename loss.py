import torch

class CrossEntropyLoss():

    # Implementation of Cross Entropy Loss

    def __init__(self):
        self.cache = ()

    def forward(self, y_pred, y_truth):

        # Assuming y_pred is of (N,C) and y_truth (N)

        N = y_pred.shape[0]
        
        # Numerical stability by subtracting the max
        logits_max = torch.max(y_pred, dim=1, keepdim=True).values
        logits_norm = y_pred - logits_max
        counts = torch.exp(logits_norm)
        counts_sum = torch.sum(counts, dim=1, keepdim = True)
        counts_sum_inv = counts_sum**-1
        probs = counts * counts_sum_inv
        pred = probs[torch.arange(N), y_truth]
        pred_logs = torch.log(pred)
        loss = -torch.mean(pred_logs)

        self.cache = (pred_logs, pred, probs, y_truth, counts, counts_sum, counts_sum_inv, y_pred)
        return loss

    def backward(self, dout):

        (pred_logs, pred, probs, y_truth, counts, counts_sum, counts_sum_inv, y_pred) = self.cache

        N = pred_logs.shape[0]
        dpred_logs = -torch.ones_like(pred_logs) * N**-1
        dpred = dpred_logs/pred
        dprobs = torch.zeros_like(probs)
        dprobs[torch.arange(N), y_truth] = dpred
        dcounts_sum_inv = torch.sum(dprobs * counts, dim = 1, keepdim = True)
        dcounts_sum = -dcounts_sum_inv*counts_sum**-2
        dcounts1 = dprobs * counts_sum_inv
        dcounts2 = dcounts_sum * torch.ones_like(counts)
        dcounts=dcounts1+dcounts2
        dlogits_norm = dcounts * counts
        dlogits_max = -torch.sum(dlogits_norm, dim = 1, keepdim= True)
        dy_pred1 = dlogits_norm.clone()

        dy_pred2 = torch.zeros_like(y_pred)
        dy_pred2[torch.arange(N), torch.argmax(y_pred, dim = 1)] = 1
        dy_pred2 = dy_pred2 * dlogits_max

        dy_pred = dy_pred1 + dy_pred2

        return dout * dy_pred

        # Backprop w.r.t y_pred can be simplified:
        # N = y_pred.shape[0]
        # dy_pred = probs.clone()
        # dy_pred[torch.arange(N), y_truth] -= 1
        # dy_pred /= N
        # return dout*dy_pred