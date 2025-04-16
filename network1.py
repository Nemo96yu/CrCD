import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


def conv_3xnxn(inp, oup, kernel_size=3, stride=3,padding=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, padding, padding))


def conv_1xnxn(inp, oup, kernel_size=3, stride=3,padding=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding))


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, (1, 5, 5), (1, 1, 1), (0, 2, 2), groups=dim)
        self.conv_spatial0 = nn.Conv3d(dim, dim, (1, 5, 5), (1, 1, 1), (0, 4, 4), groups=dim, dilation=(1, 2, 2))
        self.conv_spatial = nn.Conv3d(dim, dim, (3, 5, 5), (1, 1, 1), (1, 10, 10), groups=dim, dilation=(1, 5, 5))
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        u = x.clone()
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(x)
        attn3 = self.conv1(x)
        attn4 = self.conv_spatial0(x)
        attn = attn1 + attn2 + attn3 + attn4
        attn = self.drop(attn)

        return u * attn

class CNN3D(nn.Module):
    def __init__(self, in_chans=1, embed_dim=768):
        super(CNN3D, self).__init__()
        # 20 9 0  7 7 0
        self.conv1 = nn.Conv3d(in_chans, embed_dim//4, (7, 1, 1), (5, 1, 1), (2, 0, 0))
        self.bn1 = nn.BatchNorm3d(embed_dim//4)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(embed_dim//4, embed_dim//2, (5, 3, 3), (3, 1, 1), (1, 1, 1))
        self.bn2 = nn.BatchNorm3d(embed_dim//2)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv3d(embed_dim//2, embed_dim, (3, 3, 3), (2, 1, 1), (0, 1, 1))
        self.bn3 = nn.BatchNorm3d(embed_dim)


    def forward(self, data):

        output = self.conv1(data)
        output = self.bn1(output)
        output = self.act1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.act2(output)
        output = self.conv3(output)
        output = self.bn3(output)

        return output



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Spa_FC(nn.Module):
    def __init__(self, dim, segment_dim=8, tmp=7, C=3, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.tmp = tmp
        dim2 = segment_dim * tmp
        self.mlp_h = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_d = Mlp(384, 384//3, 384)

        # init weight problem
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W, D = x.shape

        q = D // self.segment_dim
        M= self.segment_dim
        S = H
        # H
        h = x.transpose(3, 2).reshape(B, C, H*W//S, S, q, M).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H*W//S, q, S * M)
        h = self.mlp_h(h).reshape(B, C,  H*W//S, q, S, M).permute(0, 1, 2, 4, 3, 5).reshape(B, C, W, H, D).transpose(3, 2)
        # W
        w = x.reshape(B, C, H * W//S, S, q, M).permute(0, 1, 2, 4, 3, 5).reshape(B, C,  H*W//S, q, S * M)
        w = self.mlp_w(w).reshape(B, C, H*W//S, q, S, M).permute(0, 1, 2, 4, 3, 5).reshape(B, C, W, H, D)
        # C
        c = self.mlp_c(x)
        # d
        d = x.permute(0, 2, 3, 1, 4).reshape(B, W, H, C*D)
        d = self.mlp_d(d).reshape(B, W, H, C, D).permute(0, 3, 1, 2, 4)

        a = (h + w + c).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, D, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2] + 0.2 * d


        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class PermutatorBlock(nn.Module):
    def __init__(self, dim, segment_dim, tmp, band, C, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.fc = Spa_FC(dim, segment_dim=segment_dim, tmp=tmp, C=C, qkv_bias=qkv_bias)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        # xs = x + self.s_fc(self.s_norm1(x))
        x = x + self.drop_path(self.fc(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = conv_1xnxn(in_embed_dim, out_embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_embed_dim)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)  # B, C, T, H, W
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x

class SSMLP(nn.Module):
    """ MorphMLP
    """

    def __init__(self,Patch=21, BAND=8, CLASSES_NUM=9,layers=3,embed_dims=128,segment_dim=8):
        super().__init__()
        global t_stride


        num_classes = CLASSES_NUM

        in_chans = 1
        layers = layers
        segment_dim = segment_dim
        mlp_ratios = 3
        embed_dims = embed_dims

        tmp = Patch
        qkv_bias = True
        C = int(embed_dims/segment_dim)

        drop_path_rate = 0.8
        norm_layer = nn.LayerNorm

        skip_lam = 1.0

        self.num_classes = num_classes

        self.patch_embed1 = CNN3D(in_chans=in_chans, embed_dim=embed_dims).cuda(6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]

        # stage1
        self.blocks1 = nn.ModuleList([])
        for i in range(layers):
            self.blocks1.append(
                PermutatorBlock(embed_dims, segment_dim, tmp=tmp, band=BAND, C=C, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, drop_path=dpr[i], skip_lam=skip_lam)
            )
        self.attention = LKA(128)
        self.norm = norm_layer(128)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Classifier head
        self.head = nn.Linear(128, num_classes) if num_classes > 0 else nn.Identity()
        self.head1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if 't_fc.mlp_t.weight' in name:
                nn.init.constant_(p, 0)
            if 't_fc.mlp_t.bias' in name:
                nn.init.constant_(p, 0)
            if 't_fc.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_fc.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pretrained_model(self, cfg):
        if cfg.MORPH.PRETRAIN_PATH:
            checkpoint = torch.load(cfg.MORPH.PRETRAIN_PATH, map_location='cpu')
            if self.num_classes != 1000:
                del checkpoint['head.weight']
                del checkpoint['head.bias']
            return checkpoint
        else:
            return None

    def forward_features(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        x = self.patch_embed1(x)
        x = x + 0.3 * self.avgpool(x)
        # B,C,T,H,W -> B,T,H,W,C
        x = x.permute(0, 2, 3, 4, 1)

        for blk in self.blocks1:
            x = blk(x)

        x = x.permute(0, 4, 1, 2, 3)
        x = x + self.attention(x)
        x = x.permute(0, 2, 3, 4, 1)
        B, T, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):

        x = self.forward_features(x)
        x = self.norm(x)
        x = x.mean(1)

        return self.head(x), F.normalize(self.head1(x), dim=1)