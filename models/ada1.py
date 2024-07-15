import torch
import torch.nn as nn
import torch.nn.functional as F

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

class AdaINLayer(nn.Module):
    def __init__(self, input_nc, modulation_nc):
        super().__init__()
        self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)
        nhidden = 128
        use_bias = True
        self.mlp_shared = nn.Sequential(
            nn.Linear(modulation_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

    def forward(self, input, modulation_input):
        normalized = self.InstanceNorm2d(input)
        modulation_input = modulation_input.view(modulation_input.size(0), -1)
        actv = self.mlp_shared(modulation_input)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        gamma = gamma.view(*gamma.size()[:2], 1, 1)
        beta = beta.view(*beta.size()[:2], 1, 1)
        out = normalized * (1 + gamma) + beta
        return out

class AdaIN(nn.Module):
    def __init__(self, input_channel=256, modulation_channel=256, kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2)
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

class FusedModel(nn.Module):
    def __init__(self, para_ch, feature_ch):
        super(FusedModel, self).__init__()
        self.adaat = AdaAT(para_ch, feature_ch)
        self.adain = AdaIN(input_channel=feature_ch, modulation_channel=para_ch)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable parameter to control the influence of AdaIN

    def forward(self, feature_map, para_code):
        transformed_feature_map = self.adaat(feature_map, para_code)
        adain_output = self.adain(transformed_feature_map, para_code)
        output = transformed_feature_map + adain_output * self.alpha
        return output

# Example usage:
# para_ch = 256
# feature_ch = 256
# fused_model = FusedModel(para_ch, feature_ch)
# feature_map = torch.randn(8, 256, 26, 20)
# para_code = torch.randn(8, 256)
# output = fused_model(feature_map, para_code)
