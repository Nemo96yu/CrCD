import torch
import torch.nn as nn
import torch.nn.functional as F


class networkB(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()

        self.encoder_q = base_encoder()
        self.register_buffer("queue", torch.randn(args.moco_queue, 64))
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("prototypes", torch.zeros(9, 64))  # 初始时不为零向量的随机数？
        self.queue = F.normalize(self.queue, dim=0)
        # MLP
        self.dim_in = 512
        self.feat_dim = 128
        self.head = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_in, self.feat_dim)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * (1 - args.moco_m) + param_q.data * args.moco_m

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0

        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue

        self.queue_ptr[0] = ptr

    def forward(self, img_q, partial_Y=None, args=None, eval_only=False):
        output, k = self.encoder_q(img_q)
        w =0
        if eval_only:
            return output

        # 伪标签
        predicted_scores = torch.mul(torch.softmax(output, dim=1), torch.tensor(partial_Y).cuda(6))
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)

        # 原型向量的预测标签
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(k, prototypes.t())

        score_prot = torch.softmax(logits_prot, dim=1)
        for feat_q, label in zip(k, pseudo_labels_b):
            self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat_q

        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        labels = logits_prot
        W = 0
        return output, k, score_prot, labels, prototypes, W


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output