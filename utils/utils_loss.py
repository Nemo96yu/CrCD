import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import cosine_similarity

class partial_loss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        # self.confidence = torch.tensor([item.cpu().detach().numpy() for item in confidence]).cuda()

        self.confidence1 = confidence.cuda(6)
        self.confidence2 = confidence.cuda(6)
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index, rank):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        average_loss = 0
        if rank == 1:
            final_outputs = logsm_outputs * self.confidence1[index[:, 0], :]
            average_loss = - ((final_outputs).sum(dim=1)).mean()
        if rank == 2:
            final_outputs = logsm_outputs * self.confidence2[index[:, 0], :]
            average_loss = - ((final_outputs).sum(dim=1)).mean()

        return average_loss
    
    def confidence_update(self, temp_un_conf, batch_index, batchY, rank):
        with torch.no_grad():
            # 取key的最大预测值做one_hot

            _, prot_pred = (temp_un_conf.cuda(6) * batchY.cuda(6)).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda(6).detach()
            # pseudo_label = temp_un_conf
            if rank == 1:
                self.confidence1[batch_index[:, 0], :] = self.conf_ema_m * self.confidence1[batch_index[:, 0], :] \
                                                         + (1 - self.conf_ema_m) * pseudo_label
            if rank == 2:
                self.confidence2[batch_index[:, 0], :] = self.conf_ema_m * self.confidence2[batch_index[:, 0], :] \
                                                         + (1 - self.conf_ema_m) * pseudo_label
    def get_confidence(self):
        confidence1 = self.confidence1
        confidence2 = self.confidence2
        return confidence1, confidence2


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda:8')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().cuda(6)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size * 2], features.T),
                self.temperature)
            # for numerical stability

            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * 2).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # InfoNCE loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)


            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature


            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(6)
            loss = F.cross_entropy(logits, labels)


        return loss

class SinkhornDistance(nn.Module):

    def __init__(self, eps, max_iter, reduction='sum'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).cuda(6)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda(6)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda(6)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        # return cost, pi, C
        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

