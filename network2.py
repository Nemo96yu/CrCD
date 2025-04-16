import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_


class CNN2D(nn.Module):
    def __init__(self, in_chans, embed_dims):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, embed_dims//4, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(embed_dims//4)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1)
        # self.conv2d = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(embed_dims//2)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1)
        # self.conv3d = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(embed_dims)


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

def conv_3xnxn(inp, oup, kernel_size=3, stride=3,padding=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, padding, padding))


def conv_1xnxn(inp, oup, kernel_size=3, stride=3, padding=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding))


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=10, groups=dim, dilation=5)
        self.conv1 = nn.Conv2d(dim, dim, 1)
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
        # self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_c = Mlp(128, 128//4, 128)

        # init weight problem
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, D = x.shape

        q = D // self.segment_dim
        M= self.segment_dim
        S = H
        # H
        h = x.transpose(2, 1).reshape(B, H*W//S, S, q, M).permute(0, 1, 3, 2, 4).reshape(B, H*W//S, q, S * M)
        h = self.mlp_h(h).reshape(B, H*W//S, q, S, M).permute(0, 1, 3, 2, 4).reshape(B, W, H, D).transpose(2, 1)
        # W
        w = x.reshape(B, H * W//S, S, q, M).permute(0, 1, 3, 2, 4).reshape(B, H*W//S, q, S * M)
        w = self.mlp_w(w).reshape(B, H*W//S, q, S, M).permute(0, 1, 3, 2, 4).reshape(B, W, H, D)
        # C
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, D, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]
        # x = c

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Spe_FC(nn.Module):
    def __init__(self, dim, segment_dim, band,C ,qkv_bias=False, proj_drop=0.):
        super().__init__()

        self.segment_dim =segment_dim
        dim2 = segment_dim * 3

        self.mlp_t = nn.Linear(dim2, dim2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, D= x.shape
        M = self.segment_dim
        q = D // self.segment_dim


        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class PermutatorBlock(nn.Module):
    def __init__(self, dim, segment_dim, tmp, band, C, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.s_norm1 = norm_layer(dim)
        # self.s_fc = Spe_FC(dim, segment_dim, band, C, qkv_bias=qkv_bias)
        self.fc = Spa_FC(dim, segment_dim=segment_dim, tmp=tmp, C=C, qkv_bias=qkv_bias)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.fc(self.norm1(x))) / self.skip_lam
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj1 = conv_3xnxn(in_chans, embed_dim//4, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.BatchNorm3d(embed_dim//4)
        self.act1 = nn.GELU()
        self.proj2 = conv_3xnxn(embed_dim//4, embed_dim // 2, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm3d(embed_dim // 2)
        self.act2 = nn.GELU()
        self.proj3 = conv_1xnxn(embed_dim//2, embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm3d(embed_dim)

    def forward(self, x):
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.proj2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.proj3(x)
        x = self.norm3(x)
        return x

class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = conv_1xnxn(in_embed_dim, out_embed_dim, kernel_size=1, stride=2,padding=1)
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

    def __init__(self,Patch=21, BAND=8, CLASSES_NUM=9, layers=3, embed_dims=128, segment_dim=8):
        super().__init__()
        global t_stride


        num_classes = CLASSES_NUM

        in_chans = 3
        layers = layers
        segment_dim = segment_dim
        mlp_ratios = 3
        embed_dims = embed_dims

        tmp = Patch
        qkv_bias = True
        C = int(embed_dims/segment_dim)

        drop_path_rate = 0.8
        norm_layer = nn.LayerNorm

        skip_lam = 1

        self.num_classes = num_classes

        # self.patch_embed1 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims)
        self.patch_embed1 = CNN2D(in_chans=in_chans, embed_dims=embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        # for item in dpr:
        #     print(item)

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
        # x = x.view(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])
        x = self.patch_embed1(x)
        x = x + 0.3 * self.avgpool(x)
        x = x.permute(0, 2, 3, 1)

        for blk in self.blocks1:
            x = blk(x)

        x = x.permute(0, 3, 2, 1)
        x = x + self.attention(x)
        x = x.permute(0, 3, 2, 1)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):

        x = self.forward_features(x)
        x = self.norm(x)
        x = x.mean(1)
        return self.head(x), F.normalize(self.head1(x), dim=1)