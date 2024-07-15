import torch.nn.functional as F
from torch import nn
import torch

class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=4, pool=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size
        )
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out

class DownBlock2d1(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=4, pool=False):
        super(DownBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(
            in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_features)
        self.residual = nn.Conv2d(
            in_channels=in_features, out_channels=out_features, kernel_size=1
        )
        self.pool = pool

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.max_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator for GAN loss
    """

    def __init__(
        self, num_channels, block_expansion=64, num_blocks=4, max_features=512
    ):
        super(Discriminator, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    num_channels
                    if i == 0
                    else min(max_features, block_expansion * (2**i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=4,
                    pool=(i != num_blocks - 1),
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(
            self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1
        )

    def forward(self, x):
        feature_maps = []
        out = x
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        out = self.conv(out)
        return feature_maps, out








class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size, stride, padding),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)




class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
        nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

        nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
        nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

        nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2), # 24,24
        nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

        nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # 12,12
        nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

        nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 6,6
        nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

        nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 3,3
        nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1), ),

        nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0), # 1, 1
        nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2) // 2:]

    def perceptual_forward(self, false_face_sequences):

        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1),
        torch.ones((len(false_feats), 1)).to(false_face_sequences.device)) #.cuda()

        return false_pred_loss

    def forward(self, face_sequences):

        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
    
    
    
import torch
import torch.nn as nn
from models.diff_aug import DiffAugment

class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
    dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.droprateout(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = 1./dim**0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c//self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x

class ImgPatches(nn.Module):
    def __init__(self, input_channel=3, dim=768, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channel, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches

def UpSampling(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

class Encoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.Encoder_Blocks = nn.ModuleList([
        Encoder_Block(dim, heads, mlp_ratio, drop_rate)
        for i in range(depth)])

    def forward(self, x):
        for Encoder_Block in self.Encoder_Blocks:
            x = Encoder_Block(x)
        return x

class Generator(nn.Module):
    def __init__(self, depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.):#,device=device):
        super(Generator, self).__init__()

        #self.device = device
        self.initial_size = initial_size
        self.dim = dim
        self.depth1 = depth1
        self.depth2 = depth2
        self.depth3 = depth3
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate =drop_rate

        self.mlp = nn.Linear(1024, (self.initial_size ** 2) * self.dim)

        self.positional_embedding_1 = nn.Parameter(torch.zeros(1, (8**2), 384))
        self.positional_embedding_2 = nn.Parameter(torch.zeros(1, (8*2)**2, 384//4))
        self.positional_embedding_3 = nn.Parameter(torch.zeros(1, (8*4)**2, 384//16))

        self.TransformerEncoder_encoder1 = TransformerEncoder(depth=self.depth1, dim=self.dim,heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder2 = TransformerEncoder(depth=self.depth2, dim=self.dim//4, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)
        self.TransformerEncoder_encoder3 = TransformerEncoder(depth=self.depth3, dim=self.dim//16, heads=self.heads, mlp_ratio=self.mlp_ratio, drop_rate=self.droprate_rate)


        self.linear = nn.Sequential(nn.Conv2d(self.dim//16, 3, 1, 1, 0))

    def forward(self, noise):

        x = self.mlp(noise).view(-1, self.initial_size ** 2, self.dim)

        x = x + self.positional_embedding_1
        H, W = self.initial_size, self.initial_size
        x = self.TransformerEncoder_encoder1(x)

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_2
        x = self.TransformerEncoder_encoder2(x)

        x,H,W = UpSampling(x,H,W)
        x = x + self.positional_embedding_3

        x = self.TransformerEncoder_encoder3(x)
        x = self.linear(x.permute(0, 2, 1).view(-1, self.dim//16, H, W))

        return x

class Discriminator1(nn.Module):
    def __init__(self, diff_aug, image_size=(104, 80), patch_size=4, input_channel=3, num_classes=1,
        dim=96, depth=7, heads=4, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (image_size[0]//patch_size) * (image_size[1]//patch_size)
        self.diff_aug = diff_aug
        self.patch_size = patch_size
        self.depth = depth
        # Image patches and embedding layer
        self.patches = ImgPatches(input_channel, dim, self.patch_size)

        # Embedding for patch position and class
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches+1, dim))
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.2)
        nn.init.trunc_normal_(self.class_embedding, std=0.2)

        self.droprate = nn.Dropout(p=drop_rate)
        self.TransfomerEncoder = TransformerEncoder(depth, dim, heads,
        mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = DiffAugment(x, self.diff_aug)
        b = x.shape[0]
        cls_token = self.class_embedding.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.droprate(x)
        x = self.TransfomerEncoder(x)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return x