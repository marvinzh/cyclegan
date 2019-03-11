import torch.nn as nn
import torch.nn.functional as F
import torch

import argparse
import itertools
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator
import dataset
import hparams as C
from data_utils import ReplayBuffer


def create_checkpoint():
    pass


def generator_trian_step():
    pass


def source_discriminator_train_step():
    pass


def target_discriminator_train_step():
    pass


if __name__ == "__main__":

    g_s2t = Generator(C.nc_input, C.nc_output, C.n_res_block)
    g_t2s = Generator(C.nc_input, C.nc_output, C.n_res_block)
    d_src = Discriminator(C.input_nc)
    d_trg = Discriminator(C.input_nc)

    if C.use_cuda:
        g_s2t.cuda()
        g_t2s.cuda()
        d_src.cuda()
        d_trg.cuda()

    gan_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()

    g_opt = torch.optim.Adam(
        [g_s2t.parameters(), g_t2s.parameters()], C.learning_rate, betas=(0.5, 0.999))

    d_src_opt = torch.optim.Adam(
        [d_src.parameters()], C.learning_rate, betas=(0.5, 0.999))
    d_trg_opt = torch.optim.Adam(
        [d_trg.parameters()], C.learning_rate, betas=(0.5, 0.999))

    # load data
    # source
    mixer_dataset = dataset.IVecDataset(C.MIXER_PATH)

    # target
    swbd_dataset = dataset.IVecDataset(C.SWBD_PATH)

    swbd_data = DataLoader(
        swbd_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.n_cpu)
    mixer_data = DataLoader(
        mixer_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.n_cpu)

    reals_label = torch.ones(C.batch_size)
    fakes_label = torch.zeros(C.batch_size)

    fake_src_buffer = ReplayBuffer()
    fake_trg_buffer = ReplayBuffer()

    for epoch in range(C.n_epoch):
        for i, (swbd, mixer) in enumerate(zip(swbd_data, mixer_data)):

            # identity loss
            idt_source = g_t2s(mixer)
            loss_idt_src = identity_loss(idt_source, mixer)

            idt_target = g_s2t(swbd)
            loss_idt_trg = identity_loss(idt_target, swbd)

            # gan loss
            src_fakes = g_t2s(swbd)
            preds_src_fakes = d_src(src_fakes)
            loss_gan_src = gan_loss(preds_src_fakes, reals_label)

            trg_fakes = g_s2t(mixer)
            preds_trg_fakes = d_trg(trg_fakes)
            loss_gan_trg = gan_loss(preds_trg_fakes, reals_label)

            # cycle loss
            recoverd_src = g_t2s(trg_fakes)
            loss_cyc_src = cycle_loss(recoverd_src, mixer)

            recoverd_trg = g_s2t(src_fakes)
            loss_cyc_trg = cycle_loss(recoverd_trg, swbd)

            loss_g = loss_gan_src+loss_gan_trg + C.idt_lambda * loss_idt_src + \
                C.idt_lambda * loss_idt_trg + C.cycle_gamma * \
                loss_cyc_src + C.cycle_gamma*loss_cyc_trg

            g_opt.zero_grad()
            loss_g.backward()
            g_opt.step()

            # source discriminator loss
            pred_real = d_src(mixer)
            loss_d_src_real = gan_loss(pred_real, reals_label)

            fake_src = fake_src_buffer.push_and_pop(src_fakes)
            pred_fake = d_src(fake_src.detach())
            loss_d_src_fake = gan_loss(pred_fake, fakes_label)

            loss_d_src = (loss_d_src_real + loss_d_src_fake) * 0.5
            d_src_opt.zero_grad()
            loss_d_src.backward()
            d_src_opt.step()

            # target discriminator loss

            pred_real = d_trg(swbd)
            loss_d_trg_real = gan_loss(pred_real, reals_label)

            fake_trg = fake_trg_buffer.push_and_pop(trg_fakes)
            pred_fake = d_trg(fake_trg.detach())
            loss_d_trg_fake = gan_loss(pred_fake, fakes_label)

            loss_d_trg = (loss_d_trg_real + loss_d_trg_fake) * 0.5
            d_trg_opt.zero_grad()
            loss_d_trg.backward()
            d_trg_opt.step()

            print("Iteration: %4d \nG_loss:%.6f, G_T2S_loss: %.6f, G_S2T_loss: %.6f, G_identity_loss: %.6f, G_cycle_loss: %.6f" % (
                i, loss_g.item(), loss_gan_src.item(), loss_gan_trg.item(), (loss_idt_src.item()+loss_idt_trg.item())*C.idt_lambda, C.cycle_gamma*(loss_cyc_src.item()+loss_cyc_trg.item()))
            print("D_loss %.6f" % loss_d_src+loss_d_trg)

        create_checkpoint()
