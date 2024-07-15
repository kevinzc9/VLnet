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
from models.DINet import DINet,DINet1
from models.Discriminator import Discriminator,Discriminator1
from models.VGG19 import Vgg19
from sync_batchnorm import convert_model
from utils.training_utils import GANLoss, get_scheduler, update_learning_rate

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

    # net_dI = Discriminator1('translation,cutout,color',(104,80),4).cuda()
    net_vgg = Vgg19().cuda()
    # parallel
    net_g = nn.DataParallel(net_g)
    net_g = convert_model(net_g)
    net_vgg = nn.DataParallel(net_vgg)
    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
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
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    # start train
    for epoch in range(opt.start_epoch, opt.non_decay + opt.decay + 1):
        net_g.train()
        for iteration, data in enumerate(training_data_loader):
            # start_time = time.time()
            # read data
            (
                source_image_data,
                source_image_mask,
                reference_clip_data,
                deepspeech_feature,
                source_image_mouth,
                reference_clip_data_mouth,
                reference_clip_data_without_concat
            ) = data
            # deepspeech [24,29,5]
            # reference_clip_data [24,15,104,80]
            # source_image_data [24,3,104,80]
            # source_image_mask [24,3,104,80]
            # reference_clip_data_mouth [24,3,5,104,80]
            # reference_clip_data_without_concat [24,3,5,104,80]
            source_image_data = source_image_data.float().cuda()
            source_image_mask = source_image_mask.float().cuda()
            reference_clip_data = reference_clip_data.float().cuda()
            deepspeech_feature = deepspeech_feature.float().cuda()
            reference_clip_data_without_concat = reference_clip_data_without_concat.float().cuda()
            reference_clip_data_mouth=reference_clip_data_mouth.float().cuda()
            # network forward
            fake_out = net_g(source_image_mask, reference_clip_data, deepspeech_feature,reference_clip_data_mouth,reference_clip_data_without_concat)
            # fake_out = net_g(source_image_mask, reference_clip_data, deepspeech_feature)
            # down sample output image and real image
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            target_tensor_half = F.interpolate(
                source_image_data, scale_factor=0.5, mode="bilinear"
            )

            optimizer_g.zero_grad()
            # compute perception loss
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
            # # gan dI loss
            # combine perception loss and gan loss
            loss_g = loss_g_perception 
            loss_g.backward()
            optimizer_g.step()
            # end_time = time.time()  # End time of the epoch
            # elapsed_time = end_time - start_time
            print(
                "===> Epoch[{}]({}/{}):  Loss_perception: {:.4f} lr_g = {:.7f}".format(
                    epoch,
                    iteration,
                    len(training_data_loader),
                    float(loss_g_perception),
                    optimizer_g.param_groups[0]["lr"],
                )
            )
            # print(f"elapsed time is {elapsed_time}")
        update_learning_rate(net_g_scheduler, optimizer_g)
        # checkpoint
        # if epoch % opt.checkpoint == 0:
        if epoch % 400 == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(
                opt.result_path, "netG_model_epoch_{}_Loss_perception:{:.4f}.pth".format(epoch,float(loss_g_perception))
            )
            states = {
                "epoch": epoch + 1,
                "state_dict": {
                    "net_g": net_g.state_dict(),
                },
                "optimizer": {
                    "net_g": optimizer_g.state_dict(),
                },
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))
