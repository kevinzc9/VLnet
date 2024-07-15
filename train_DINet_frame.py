import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config.config import DINetTrainingOptions
from dataset.dataset_DINet_frame import DINetDataset
from models.DINet import DINet,DINet1,DINet3,DINet_best,DINet_origin
from models.Discriminator import Discriminator,Discriminator1
from models.VGG19 import Vgg19
from sync_batchnorm import convert_model
from utils.training_utils import GANLoss, get_scheduler, update_learning_rate

from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import lpips
class WGANGPDiscriminatorLoss(nn.Module):
    def __init__(self, lambda_gp=10.0):
        super(WGANGPDiscriminatorLoss, self).__init__()
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, discriminator, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1).cuda()
        interpolates = (epsilon * real_samples + ((1 - epsilon) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)[1]
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty

    def forward(self, discriminator, real_samples, fake_samples):
        real_validity = discriminator(real_samples)[1]
        fake_validity = discriminator(fake_samples)[1]
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        gp = self.gradient_penalty(discriminator, real_samples, fake_samples)
        d_loss += gp
        return d_loss

class WGANGLoss(nn.Module):
    def __init__(self):
        super(WGANGLoss, self).__init__()

    def forward(self, fake_validity):
        return -torch.mean(fake_validity)




if __name__ == "__main__":
    """
    frame training code of DINet
    we use coarse-to-fine training strategy
    so you can use this code to train the model in arbitrary resolution
    """
    # load config
    opt = DINetTrainingOptions().parse_args()
    # set seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data in memory
    train_data = DINetDataset(opt.train_data, opt.augment_num, opt.mouth_region_size)
    training_data_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    train_data_length = len(training_data_loader)
    # init network
    net_g = DINet1(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    net_dI = Discriminator(
        opt.source_channel, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features
    ).cuda()
    net_vgg = Vgg19().cuda()
    # parallel
    net_g = nn.DataParallel(net_g)
    net_g = convert_model(net_g)
    net_dI = nn.DataParallel(net_dI)
    net_vgg = nn.DataParallel(net_vgg)
    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)
    # coarse2fine
    if opt.coarse2fine:
        print(
            "loading checkpoint for coarse2fine training: {}".format(
                opt.coarse_model_path
            )
        )
        checkpoint = torch.load(opt.coarse_model_path)
        net_g.load_state_dict(checkpoint["state_dict"]["net_g"])
    # set criterion
    # criterionGAN = GANLoss().cuda()
    criterionD = WGANGPDiscriminatorLoss().cuda()
    criterionG = WGANGLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)

    # 初始化 SSIM, PSNR, and LPIPS 计算器
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    lpips_metric = lpips.LPIPS(net='alex').cuda()  # 使用 AlexNet 作为 backbone

    # start train
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        net_g.train()
        total_ssim = 0.0
        total_psnr = 0.0
        total_lpips = 0.0
        for iteration, data in enumerate(training_data_loader):
            # read data
            (
                source_image_data,
                source_image_mask,
                reference_clip_data,
                deepspeech_feature
            ) = data
            source_image_data = source_image_data.float().cuda()
            source_image_mask = source_image_mask.float().cuda()
            reference_clip_data = reference_clip_data.float().cuda()
            deepspeech_feature = deepspeech_feature.float().cuda()
            
            # network forward
            fake_out = net_g(source_image_mask, reference_clip_data, deepspeech_feature)
            
            # down sample output image and real image
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            target_tensor_half = F.interpolate(
                source_image_data, scale_factor=0.5, mode="bilinear"
            )
            
            # (1) Update D network
            optimizer_dI.zero_grad()
            _, pred_fake_dI = net_dI(fake_out)
            
            # loss_dI_fake = criterionGAN(pred_fake_dI, False)
            _, pred_real_dI = net_dI(source_image_data)
            # loss_dI_real = criterionGAN(pred_real_dI, True)
            # loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
            loss_dI = criterionD(net_dI, source_image_data, fake_out)
            loss_dI.backward(retain_graph=True)
            optimizer_dI.step()
            
            # (2) Update G network
            _, pred_fake_dI = net_dI(fake_out)
            optimizer_g.zero_grad()
            perception_real = net_vgg(source_image_data)
            perception_fake = net_vgg(fake_out)
            perception_real_half = net_vgg(target_tensor_half)
            perception_fake_half = net_vgg(fake_out_half)
            loss_g_perception = 0
            for i in range(len(perception_real)):
                loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                loss_g_perception += criterionL1(
                    perception_fake_half[i], perception_real_half[i]
                )
            loss_g_perception = (
                loss_g_perception / (len(perception_real) * 2)
            ) * opt.lamb_perception
            # loss_g_dI = criterionGAN(pred_fake_dI, True)
            loss_g_dI = criterionG(pred_fake_dI)

            # Add L1 reconstruct loss
            loss_g_L1 = criterionL1(fake_out, source_image_data) * opt.lamb_L1

            # Combine all G losses
            loss_g = loss_g_perception + loss_g_dI + loss_g_L1
            loss_g.backward()
            optimizer_g.step()

            # Calculate metrics
            ssim_value = ssim_metric(fake_out, source_image_data)
            psnr_value = psnr_metric(fake_out, source_image_data)
            lpips_value = lpips_metric(fake_out, source_image_data).mean().item()
            total_ssim += ssim_value.item()
            total_psnr += psnr_value.item()
            total_lpips += lpips_value

            print(
                "===> Epoch[{}]({}/{}):  Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_perception: {:.4f} Loss_L1: {:.4f} SSIM: {:.4f} PSNR: {:.4f} LPIPS: {:.4f} lr_g = {:.7f}".format(
                    epoch,
                    iteration,
                    len(training_data_loader),
                    float(loss_dI),
                    float(loss_g_dI),
                    float(loss_g_perception),
                    float(loss_g_L1),
                    ssim_value.item(),
                    psnr_value.item(),
                    lpips_value,
                    optimizer_g.param_groups[0]["lr"],
                )
            )

        # Average metrics
        avg_ssim = total_ssim / len(training_data_loader)
        avg_psnr = total_psnr / len(training_data_loader)
        avg_lpips = total_lpips / len(training_data_loader)
        
        print(f"===> adaatmlplearn lab Epoch [{epoch}] Complete: Avg SSIM: {avg_ssim:.4f}, Avg PSNR: {avg_psnr:.4f}, Avg LPIPS: {avg_lpips:.4f}")

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        
        # checkpoint
        if epoch % 100 == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(
                opt.result_path, "adaatmlplearn_model_epoch_{}_ssim:{:.4f}_psnr:{:.4f}_lpips:{:.4f}_Loss_perception:{:.4f}.pth".format(epoch,float(avg_ssim),float(avg_psnr),float(avg_lpips),float(loss_g_perception))
            )
            states = {
                "epoch": epoch + 1,
                "state_dict": {
                    "net_g": net_g.state_dict(),
                    "net_dI": net_dI.state_dict(),
                },
                "optimizer": {
                    "net_g": optimizer_g.state_dict(),
                    "net_dI": optimizer_dI.state_dict(),
                },
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))
