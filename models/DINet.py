import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sync_batchnorm import SynchronizedBatchNorm1d as BatchNorm1d
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from models.FusionNet import FusionNet
from models.ada import adaModel
from models.ada1 import FusedModel as adaModel1
from models.ada1 import FusedModel as adaModel2

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Hybird_transformer_encoder(nn.Module):
    def __init__(self,T=5, d_model=512, nlayers=4, nhead=4, dim_feedforward=512,  # 1024   128
                 dropout=0.1):
        super().__init__()
        self.T=T
        self.position_v = PositionalEmbedding(d_model=512)  #for visual landmarks
        self.position_a = PositionalEmbedding(d_model=512)  #for audio embedding
        self.modality = nn.Embedding(4, 512, padding_idx=0)  # 1 for pose,  2  for  audio, 3 for reference landmarks
        self.dropout = nn.Dropout(p=dropout)
        encoder_layers = TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 4)

    def forward(self,ref_embedding,mel_embedding,pose_embedding):#(B,Nl,512)  (B,T,512)    (B,T,512)

        # (1).  positional(temporal) encoding
        position_v_encoding = self.position_v(pose_embedding)  # (1,  T, 512)
        position_a_encoding = self.position_a(mel_embedding)

        #(2)  modality encoding
        modality_v = self.modality(1 * torch.ones((pose_embedding.size(0), self.T), dtype=torch.int).cuda())
        modality_a = self.modality(2 * torch.ones((mel_embedding.size(0),  self.T), dtype=torch.int).cuda())

        pose_tokens = pose_embedding + position_v_encoding + modality_v    #(B , T, 512 )
        audio_tokens = mel_embedding + position_a_encoding + modality_a    #(B , T, 512 )
        ref_tokens = ref_embedding + self.modality(
            3 * torch.ones((ref_embedding.size(0), ref_embedding.size(1)), dtype=torch.int).cuda())

        #(3) concat tokens
        input_tokens = torch.cat((ref_tokens, audio_tokens, pose_tokens), dim=1)  # (B, 1+T+T, 512 )
        input_tokens = self.dropout(input_tokens)

        #(4) input to transformer
        output = self.transformer_encoder(input_tokens)
        return output



class Converter(nn.Module):
    def __init__(self):
        super(Converter, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*3, 256)

    def forward(self, x):
        # x: [batch_size, 15, 512]
        x = x.permute(0, 2, 1)  # [batch_size, 512, 15]
        x = self.global_pooling(x)  # [batch_size, 512, 1]
        x = x.squeeze(-1)  # [batch_size, 512]
        x = self.fc(x)  # [batch_size, 256]
        return x



def make_coordinate_grid_3d(spatial_size, type):
    """
    generate 3D coordinate grid
    """
    d, h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    z = torch.arange(d).type(type)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    z = 2 * (z / (d - 1)) - 1
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed, zz



class AdaATIN(nn.Module):
    def __init__(self, para_ch, feature_ch, modulation_ch):
        super(AdaATIN, self).__init__()
        self.adaat = AdaAT(para_ch, feature_ch)
        self.adain = AdaIN(feature_ch, modulation_ch)

    def forward(self, feature_map, para_code, style_code):
        trans_feature = self.adaat(feature_map, para_code)
        styled_feature = self.adain(trans_feature, style_code)
        return styled_feature



class ResBlock1d(nn.Module):
    """
    basic block
    """

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features, out_features, 1)
        self.norm1 = BatchNorm1d(in_features)
        self.norm2 = BatchNorm1d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out


class ResBlock2d(nn.Module):
    """
    basic block
    """

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features, out_features, 1)
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        return out


class UpBlock2d(nn.Module):
    """
    basic block
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock1d(nn.Module):
    """
    basic block
    """

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(DownBlock1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=2,
        )
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    basic block
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class SameBlock1d(nn.Module):
    """
    basic block
    """

    def __init__(self, in_features, out_features, kernel_size, padding):
        super(SameBlock1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class SameBlock2d(nn.Module):
    """
    basic block
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


    
class AdaAT(nn.Module):
    """
    AdaAT operator
    """

    def __init__(self, para_ch, feature_ch):
        super(AdaAT, self).__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(nn.Linear(para_ch, para_ch), nn.ReLU())
        self.scale = nn.Sequential(nn.Linear(para_ch, feature_ch), nn.Sigmoid())
        self.rotation = nn.Sequential(nn.Linear(para_ch, feature_ch), nn.Tanh())
        self.translation = nn.Sequential(nn.Linear(para_ch, 2 * feature_ch), nn.Tanh())
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_map, para_code):
        batch, d, h, w = (
            feature_map.size(0),
            feature_map.size(1),
            feature_map.size(2),
            feature_map.size(3),
        )
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159  #
        rotation_matrix = torch.cat(
            [torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)],
            -1,
        )
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map.type())
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        rotation_matrix = (
            rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        )
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        trans_grid = (
            torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale
            + translation
        )
        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        trans_feature = F.grid_sample(feature_map.unsqueeze(1), full_grid).squeeze(1)
        return trans_feature


class DINet(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):
        super(DINet, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1),
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid(),
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)




    def forward(self, source_img, ref_img, audio_feature):
        ## source image encoder
        source_in_feature = self.source_in_conv(source_img)
        ## reference image encoder
        ref_in_feature = self.ref_in_conv(ref_img)
        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        ## audio encoder
        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        ## concat alignment feature and audio feature
        trans_para = torch.cat([img_para, audio_para], 1)
        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        s1=source_in_feature.shape
        s2=ref_trans_feature.shape
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        
        out = self.out_conv(merge_feature)
        return out
class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim

        assert feature_dim % num_heads == 0

        self.depth = feature_dim // num_heads

        self.wq = nn.Linear(feature_dim, feature_dim)
        self.wk = nn.Linear(feature_dim, feature_dim)
        self.wv = nn.Linear(feature_dim, feature_dim)

        self.dense = nn.Linear(feature_dim, 512)  # 修改这里，使输出特征维度为原来的2倍
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, query, key, value):

        batch_size = query.shape[0]

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.depth)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim * 2)  # 修改这里，使输出特征维度为原来的2倍
        
        output = output.view(batch_size, -1, 256)  # 将 output 的形状从 [24, 512, 260] 改为 [24, 512*260, 260]
        output = self.dense(output)  # 现在 output 的形状应该是 [24, 512, 520]
        output = output.view(batch_size, -1, 26, 20) 

        return output, attention_weights
    
    
    
class ave(nn.Module):
    def __init__(self, dim_in=29, dim_out=512, seq_len=5, dim_aud=128):
        super(ave, self).__init__()
        self.dim_aud = dim_aud
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(dim_out, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )
        self.fc2 = nn.Linear(seq_len, 2)

    def forward(self, x):
        batch_size, dim_in,seq_len = x.size()
        x = self.fc1(x.view(batch_size*seq_len, dim_in))
        x = self.encoder_fc1(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.fc2(x.permute(0, 2, 1))
        return x    
    

class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=128, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=29, seq_len=5):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [24, 29, 5]
        y=x
        batch_size, dim_in,seq_len = x.size()
        y = self.attentionConvNet(y).permute(0, 2, 1) # [24, 5, 128]
        attn_weights = self.attentionNet(y) # [24, 5, 128]
        y = torch.sum(y * attn_weights, dim=1) # [24, 128]
        return y


class AudioLabNet(nn.Module):
    def __init__(self, dim_aud=29, seq_len=5):
        super(AudioLabNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(128, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [24, 29, 5]
        y=x
        batch_size, dim_in,seq_len = x.size()
        y = self.attentionConvNet(y).permute(0, 2, 1) # [24, 5, 128]
        attn_weights = self.attentionNet(y) # [24, 5, 128]
        y = y * attn_weights
        return y

class AdaINLayer(nn.Module):
    def __init__(self, input_nc, modulation_nc):
        super().__init__()

        self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(modulation_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

    def forward(self, input, modulation_input):

        # Part 1. generate parameter-free normalized activations
        normalized = self.InstanceNorm2d(input)

        # Part 2. produce scaling and bias conditioned on feature
        modulation_input = modulation_input.view(modulation_input.size(0), -1)
        actv = self.mlp_shared(modulation_input)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(15, 128, kernel_size=1)
        self.upsample = nn.Upsample((104, 80))  # H and W are the desired height and width of the output image
        self.conv2 = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)  # output shape: [24, 128, 512]
        x = x.unsqueeze(-1)  # output shape: [24, 128, 512, 1]
        x = self.upsample(x)  # output shape: [24, 128, H, W]
        x = self.conv2(x)  # output shape: [24, 3, H, W]
        x = torch.sigmoid(x)  # normalize to (0, 1)
        return x


class AdaIN(torch.nn.Module):

    def __init__(self, input_channel=15, modulation_channel=256,kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel)
        self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel)

    def forward(self, x, modulation):

        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)

        return x


class ConvNet11(nn.Module):
    def __init__(self):
        super(ConvNet11, self).__init__()
        self.source_in_conv = nn.Sequential(
        SameBlock2d(3, 64, kernel_size=7, padding=3),
        DownBlock2d(64, 128, kernel_size=3, padding=1),
        DownBlock2d(128, 256, kernel_size=3, padding=1),
        SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        # Add an adaptive average pooling layer here
        self.adaptive_pool = nn.AdaptiveAvgPool2d((26, 20)) # the output size (H, W) needs to be determined by the size before the fc layer
        self.fc = nn.Linear(256 * 26 * 20, 512) # Fully connected layer to convert the output to [batchsize, T, 512]

    def forward(self, x):
        batch_size, T, _, _, _ = x.size()
        x = x.reshape(batch_size * T, 3, x.size(3), x.size(4))
        out = self.source_in_conv(x)
        out = self.adaptive_pool(out) # Add the adaptive pooling step here
        out = out.reshape(batch_size, T, -1)
        out = self.fc(out)
        return out



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(3, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        self.fc = nn.Linear(256 * 104 * 5  , 512)  # Fully connected layer to convert the output to [batchsize, T, 512]

    def forward(self, x):
        batch_size, T, _, _, _ = x.size()
        x = x.reshape(batch_size * T, 3, 104 , 80)
        out = self.source_in_conv(x)
        out = out.view(batch_size, T, -1)
        out = self.fc(out)
        return out
class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False,act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        if act=='ReLU':
            self.act = nn.ReLU()
        elif act=='Tanh':
            self.act =nn.Tanh()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
    
    

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='ReLU',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()

        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)    
import torch
import torch.nn as nn

class Converter1(nn.Module):
    def __init__(self):
        super(Converter1, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(5 * 1536, 256)

    def forward(self, x):
        # x: [batch_size, 5, 1536]
        x = x.view(x.size(0), -1)  # 将最后两个维度展平为一个维度，变成 [batch_size, 5 * 1536]
        x = self.fc(x)  # 全连接层，将输入维度为5 * 1536的特征转换为维度为256的特征，得到 [batch_size, 256]
        return x    

class Converter2(nn.Module):
    def __init__(self):
        super(Converter2, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*3, 256)

    def forward(self, x):
        # x: [batch_size, 15, 512]
        x = x.permute(0, 2, 1)  # [batch_size, 512, 15]
        x = self.global_pooling(x)  # [batch_size, 512, 1]
        x = x.squeeze(-1)  # [batch_size, 512]
        x = self.fc(x)  # [batch_size, 256]
        return x




class DINet1(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):
        super(DINet1, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1),
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        # self.adaAT = AdaIN(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid(),
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        self.adain=adaModel2(256,256)
        self.audio_encoder = AudioAttNet()
        self.fusionnet1=FusionNet(choice=4, global_dim=256, local_dim=256)
        self.conv11=IdentityConv1x1(512,512)
        self.cross= BiModalAttention(128,64)
        # self.decoder=EnhancedDecoder()
    # def forward(self, source_img, ref_img, audio_feature,reference_clip_data_masked_mouth,reference_clip_data_without_concat):
    def forward(self, source_img, ref_img, audio_feature):
        ### stage1
        source_in_feature = self.source_in_conv(source_img)
        b=source_in_feature.size(0)
        h=source_in_feature.size(2)
        w=source_in_feature.size(3)
        source_in_feature1=source_in_feature.reshape(b,256,-1).transpose(1,2)
        
        # reference image encoder
        ref_in_feature = self.ref_in_conv(ref_img)
        ref_in_feature1=ref_in_feature.reshape(b,256,-1).transpose(1,2)
        out=self.fusionnet1(source_in_feature1,ref_in_feature1)
        out = out.reshape(b,512,h,w)
        out=self.conv11(out)
        img_para=self.trans_conv(out)
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        #[b,128,2]
        audio_para = self.audio_encoder(audio_feature)
        # [b,128]
        # trans_para=torch.cat([img_para, audio_para], 1)
        trans_para=self.cross(img_para, audio_para)
        # trans_para [b,256]
        
        
        ### start stage2
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # [24,256,26,20] [24,256]
        ref_trans_feature = self.adain(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1) #[b 512 26 20]
        out = self.out_conv(merge_feature)
        # out = self.decoder(merge_feature)
        return out
    
    
class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B X N X C
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  # B X C X N
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = torch.softmax(energy, dim=-1)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

class EnhancedDecoder(nn.Module):
    def __init__(self):
        super(EnhancedDecoder, self).__init__()
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 64, kernel_size=3, padding=1),
            UpBlock2d(64, 64, kernel_size=3, padding=1),
            Attention(64),
            ResBlock2d(64, 64, kernel_size=3, padding=1),
            UpBlock2d(64, 64, kernel_size=3, padding=1),
            Attention(64),
            nn.Conv2d(64, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.out_conv(x)
 
    
    
    
    
    
    
    
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiModalAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(BiModalAttention, self).__init__()
        self.query_fc = nn.Linear(feature_dim, attention_dim)
        self.key_fc = nn.Linear(feature_dim, attention_dim)
        self.value_fc = nn.Linear(feature_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out_fc = nn.Linear(attention_dim, feature_dim)

    def forward(self, audio_features, image_features):
        # Audio as query, image as key and value
        query_audio = self.query_fc(audio_features)
        key_image = self.key_fc(image_features)
        value_image = self.value_fc(image_features)
        
        attention_weights_audio = self.softmax(torch.bmm(query_audio.unsqueeze(1), key_image.unsqueeze(2)))
        attended_audio = torch.bmm(attention_weights_audio, value_image.unsqueeze(1)).squeeze(1)
        
        # Image as query, audio as key and value
        query_image = self.query_fc(image_features)
        key_audio = self.key_fc(audio_features)
        value_audio = self.value_fc(audio_features)
        
        attention_weights_image = self.softmax(torch.bmm(query_image.unsqueeze(1), key_audio.unsqueeze(2)))
        attended_image = torch.bmm(attention_weights_image, value_audio.unsqueeze(1)).squeeze(1)
        
        # Combine attended features
        attended_audio = self.out_fc(attended_audio)
        attended_image = self.out_fc(attended_image)
        
        fused_features = torch.cat([attended_audio, attended_image], dim=-1)
        return fused_features
    




class DINet2(nn.Module):
    def __init__(self, source_channel,ref_channel,audio_channel):
        super(DINet2, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel,64,kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128,256,kernel_size=3, padding=1)
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),

        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1)
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128,128,kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
    def forward(self, source_img,ref_img,audio_feature):
        ## source image encoder
        source_in_feature = self.source_in_conv(source_img) #24 256 26 20
        ## reference image encoder
        ref_in_feature = self.ref_in_conv(ref_img) #24 256 26 20
        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature,ref_in_feature],1)) #24 512 26 20
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        ## audio encoder
        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        ## concat alignment feature and audio feature
        trans_para = torch.cat([img_para,audio_para],1)
        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        ## feature decoder
        merge_feature = torch.cat([source_in_feature,ref_trans_feature],1)
        out = self.out_conv(merge_feature)
        return out











class Fusion_transformer_encoder(nn.Module):
    def __init__(self, d_model, nlayers, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.modality = nn.Embedding(3, d_model, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(2 * d_model, 512)  # Fully connected layer to expand dimension
        encoder_layers = TransformerEncoderLayer(512, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer to reduce dimension

    def forward(self, embedding1, embedding2):
        input_tokens = torch.cat((embedding1, embedding2), dim=1)
        input_tokens = self.fc1(input_tokens)  # Expand dimension
        input_tokens = self.dropout(input_tokens)

        output = self.transformer_encoder(input_tokens.unsqueeze(1))
        output = self.fc2(output.squeeze(1))  # Reduce dimension

        return output



class DINet3(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):
        super(DINet3, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1),
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid(),
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        self.img_model1=ConvNet11()
        self.img_model2=ConvNet11()
        self.audio_model = AudioLabNet()
        self.Ht=Hybird_transformer_encoder()
        self.cv=Converter2()
        self.Norm=nn.LayerNorm(512)
        # self.transformer = Fusion_transformer_encoder(d_model=128, nlayers=6, nhead=8, dim_feedforward=512, dropout=0.1)
        self.model = AudioAttNet()
        self.FusionNet1=FusionNet(choice=1, global_dim=512, local_dim=512)
    def forward(self, source_img, ref_img, audio_feature,reference_clip_data_masked_mouth,reference_clip_data_without_concat):
        # reference_clip_data_masked_mouth=reference_clip_data_masked_mouth.transpose(1,2)
        reference_clip_data_without_concat=reference_clip_data_without_concat.transpose(1, 2)
        ref_embedding=self.img_model1(reference_clip_data_without_concat)#[24,5,3,104,80]-->[24,5,512]
        mel_embedding=self.audio_model(audio_feature)#[24,29,5]-->[24,5,512]
        pose_embedding=self.img_model2(reference_clip_data_masked_mouth)#[24,5,3,104,80]-->[24,5,512]
        mel_embedding = self.Norm(mel_embedding) 
        pose_embedding =self.Norm(pose_embedding)   
        ref_embedding = self.Norm(ref_embedding)
        out=self.FusionNet1(mel_embedding,ref_embedding)
        out = torch.cat([out,pose_embedding],dim=1)
        # out=self.fusionnet2(out,ref_embedding)
        trans_para = self.cv(out)
        # (24,5,512) (24,5,512) (24,5,512)
        # (3,15,512) (3,15,512) (3,15,512)
        # out=self.Ht(ref_embedding,mel_embedding,pose_embedding)
        # out=torch.cat([ref_embedding, mel_embedding,pose_embedding], 1)
        # out [24,15,512]
        #audio_feature:[b,29,5] 5:sequence
        source_in_feature = self.source_in_conv(source_img)
        # reference image encoder
        ref_in_feature = self.ref_in_conv(ref_img)
        ## alignment encoder
        # img_para = self.trans_conv(torch.cat([source_in_feature, ref_in_feature], 1))
        # img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        ## audio encoder
        #[b,128,2]
        # audio_para = self.model(audio_feature)
        # [b,128]
        # trans_para=self.transformer(img_para,audio_para)
        # trans_para=self.cv(out)
        # trans_para = torch.mean(torch.stack([trans_para1, trans_para2]), dim=0)
        # trans_para=torch.cat([img_para, audio_para], 1)
        # trans_para [24,256]
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        # [15,256,104,80] [3,256]
        # [24,256,26,20] [24,256]
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        # source_in_feature=source_in_feature.view(24,256,-1).transpose(1, 2)
        # ref_trans_feature=ref_trans_feature.view(24,256,-1).transpose(1, 2)
        # merge_feature=self.FusionNet1(source_in_feature,ref_trans_feature)
        # merge_feature=merge_feature.transpose(1, 2).view(24,512,26,20)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1) #[b 512 26 20]
        out = self.out_conv(merge_feature)
        
        return out
    
    
    
class DINet_best(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):
        super(DINet_best, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1),
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid(),
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        self.model = AudioAttNet()
        self.fusionnet1=FusionNet(choice=4, global_dim=256, local_dim=256)
    def forward(self, source_img, ref_img, audio_feature,reference_clip_data_masked_mouth,reference_clip_data_without_concat):
        source_in_feature = self.source_in_conv(source_img)
        b=source_in_feature.size(0)
        h=source_in_feature.size(2)
        w=source_in_feature.size(3)
        source_in_feature1=source_in_feature.reshape(b,256,-1).transpose(1,2)
        ref_in_feature = self.ref_in_conv(ref_img)
        ref_in_feature1=ref_in_feature.reshape(b,256,-1).transpose(1,2)
        out=self.fusionnet1(source_in_feature1,ref_in_feature1)
        out = out.reshape(b,512,h,w)
        img_para=self.trans_conv(out)
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        audio_para = self.model(audio_feature)
        trans_para=torch.cat([img_para, audio_para], 1)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1) #[b 512 26 20]
        out = self.out_conv(merge_feature)
        
        return out

    
    
class DINet_test(nn.Module):
    def __init__(self, source_channel, ref_channel, audio_channel):
        super(DINet_test, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
            SameBlock2d(256, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1),
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid(),
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        self.img_model1=ConvNet11()
        self.img_model2=ConvNet11()
        self.audio_model = AudioLabNet()
        self.Ht=Hybird_transformer_encoder()
        self.cv=Converter()
        self.Norm=nn.LayerNorm(512)
        self.model = AudioAttNet()
        self.fusionnet1=FusionNet(choice=4, global_dim=512, local_dim=512)
        self.fusionnet2=FusionNet(choice=4, global_dim=1024, local_dim=512)
    def forward(self, source_img, ref_img, audio_feature,reference_clip_data_masked_mouth,reference_clip_data_without_concat):
        
        
        reference_clip_data_masked_mouth=reference_clip_data_masked_mouth.transpose(1,2)
        reference_clip_data_without_concat=reference_clip_data_without_concat.transpose(1, 2)
        ref_embedding=self.img_model1(reference_clip_data_without_concat)#[24,5,3,104,80]-->[24,5,512]
        mel_embedding=self.audio_model(audio_feature)#[24,29,5]-->[24,5,512]
        pose_embedding=self.img_model2(reference_clip_data_masked_mouth)#[24,5,3,104,80]-->[24,5,512]
        mel_embedding = self.Norm(mel_embedding) 
        pose_embedding =self.Norm(pose_embedding)   
        ref_embedding = self.Norm(ref_embedding)
        out=self.fusionnet1(mel_embedding,pose_embedding)
        out=self.fusionnet2(out,ref_embedding)
        trans_para=self.cv(out)
        source_in_feature = self.source_in_conv(source_img)
        ref_in_feature = self.ref_in_conv(ref_img)
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1) #[b 512 26 20]
        out = self.out_conv(merge_feature)
        
        return out
    
    
class DINet_origin(nn.Module):
    def __init__(self, source_channel,ref_channel,audio_channel):
        super(DINet_origin, self).__init__()
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel,64,kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128,256,kernel_size=3, padding=1)
        )
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            SameBlock2d(128, 128, kernel_size=11, padding=5),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),

        )
        self.audio_encoder = nn.Sequential(
            SameBlock1d(audio_channel, 128, kernel_size=5, padding=2),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            ResBlock1d(128, 128, 3, 1),
            DownBlock1d(128, 128, 3, 1),
            SameBlock1d(128, 128, kernel_size=3, padding=1)
        )

        appearance_conv_list = []
        for i in range(2):
            appearance_conv_list.append(
                nn.Sequential(
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                    ResBlock2d(256, 256, 3, 1),
                )
            )
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        self.adaAT = AdaAT(256, 256)
        self.out_conv = nn.Sequential(
            SameBlock2d(512, 128, kernel_size=3, padding=1),
            UpBlock2d(128,128,kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=(7, 7), padding=(3, 3)),
            nn.Sigmoid()
        )
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
    def forward(self, source_img,ref_img,audio_feature):
        ## source image encoder
        source_in_feature = self.source_in_conv(source_img)
        ## reference image encoder
        ref_in_feature = self.ref_in_conv(ref_img)
        ## alignment encoder
        img_para = self.trans_conv(torch.cat([source_in_feature,ref_in_feature],1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        ## audio encoder
        audio_para = self.audio_encoder(audio_feature)
        audio_para = self.global_avg1d(audio_para).squeeze(2)
        ## concat alignment feature and audio feature
        trans_para = torch.cat([img_para,audio_para],1)
        ## use AdaAT do spatial deformation on reference feature maps
        ref_trans_feature = self.appearance_conv_list[0](ref_in_feature)
        ref_trans_feature = self.adaAT(ref_trans_feature, trans_para)
        ref_trans_feature = self.appearance_conv_list[1](ref_trans_feature)
        ## feature decoder
        merge_feature = torch.cat([source_in_feature,ref_trans_feature],1)
        out = self.out_conv(merge_feature)
        return out
    
    

    
    
class IdentityConv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityConv1x1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x