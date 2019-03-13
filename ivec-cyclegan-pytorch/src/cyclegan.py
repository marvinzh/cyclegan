import torch.nn as nn
import torch.nn.functional as F
import torch

import argparse
import itertools
from torch.utils.data import DataLoader
import os

from generator import Generator
from discriminator import Discriminator
import dataset
import hparams as C
from data_utils import ReplayBuffer

fake_src_buffer = ReplayBuffer()
fake_trg_buffer = ReplayBuffer()


def create_checkpoint(g_s2t, g_t2s, d_src, d_trg, epoch):
    checkpoint_name = "model.%s.tar" % str(epoch)
    PATH = os.path.join(C.EXP_DIR, C.TAG, checkpoint_name)

    # torch.save({
    #     "g_s2t": g_s2t.state_dict(),
    #     "g_t2s": g_t2s.state_dict(),
    #     "d_src": d_src.state_dict(),
    #     "d_trg": d_trg.state_dict(),
    # }, PATH)
    print("Save checkpoint on %s" % PATH)


def generator_trian_step(g_s2t, g_t2s, d_src, d_trg, src_data, trg_data, gan_loss, identity_loss, cycle_loss, optim):
    reals_label = torch.ones(C.batch_size)
    reals_label = reals_label.cuda()

    # print(src_data.type(), trg_data.type())

    # identity loss
    identity_src = g_t2s(src_data)
    loss_idt_src = identity_loss(identity_src, src_data)

    identity_trg = g_s2t(trg_data)
    loss_idt_trg = identity_loss(identity_trg, trg_data)

    # gan loss
    fakes_trg = g_s2t(src_data)
    preds_fakes_trg = d_trg(fakes_trg)
    loss_gan_s2t = gan_loss(preds_fakes_trg, reals_label)

    fakes_src = g_t2s(trg_data)
    preds_fakes_src = d_src(fakes_src)
    loss_gan_t2s = gan_loss(preds_fakes_src, reals_label)

    fake_src_buffer.push(fakes_src)
    fake_trg_buffer.push(fakes_trg)

    # cycle loss
    recoverd_src = g_t2s(fakes_trg)
    loss_cyc_src = cycle_loss(recoverd_src, src_data)

    recoverd_trg = g_s2t(fakes_src)
    loss_cyc_trg = cycle_loss(recoverd_trg, trg_data)

    loss_gan = loss_gan_s2t+loss_gan_t2s
    loss_idt = C.idt_lambda * (loss_idt_src + loss_idt_trg)
    loss_cyc = C.cycle_gamma * (loss_cyc_src + loss_cyc_trg)
    loss_g = loss_gan + loss_idt + loss_cyc

    optim.zero_grad()
    loss_g.backward()
    optim.step()

    return loss_g, loss_gan_s2t, loss_gan_t2s, loss_idt_src, loss_idt_trg, loss_cyc_src, loss_cyc_trg


def discriminator_train_step(d, data, fake_buffer, loss, optim):
    reals_label = torch.ones(C.batch_size)
    fakes_label = torch.zeros(C.batch_size)

    reals_label = reals_label.cuda()
    fakes_label = fakes_label.cuda()

    pred_real = d(data)
    loss_d_real = loss(pred_real, reals_label)

    fake_src = fake_buffer.pop(C.batch_size)

    pred_fake = d(fake_src.detach())
    loss_d_fake = loss(pred_fake, fakes_label)

    loss_d = (loss_d_real + loss_d_fake) * 0.5

    optim.zero_grad()
    loss_d.backward()
    optim.step()
    return loss_d

def validate_step():
    psss

if __name__ == "__main__":

    g_s2t = Generator(C.nc_input, C.nc_output, C.n_res_block)
    g_t2s = Generator(C.nc_input, C.nc_output, C.n_res_block)
    d_src = Discriminator(C.nc_input)
    d_trg = Discriminator(C.nc_input)

    if C.use_cuda:
        g_s2t.cuda()
        g_t2s.cuda()
        d_src.cuda()
        d_trg.cuda()

    gan_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()

    g_opt = torch.optim.Adam(itertools.chain(
        g_s2t.parameters(), g_t2s.parameters()), C.learning_rate, betas=(0.5, 0.999))

    d_src_opt = torch.optim.Adam(
        d_src.parameters(), C.learning_rate, betas=(0.5, 0.999))
    d_trg_opt = torch.optim.Adam(
        d_trg.parameters(), C.learning_rate, betas=(0.5, 0.999))

    # load data
    # source
    mixer_dataset = dataset.IVecDataset(C.MIXER_PATH)

    # target
    swbd_dataset = dataset.IVecDataset(C.SWBD_PATH)

    swbd_data = DataLoader(
        swbd_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.n_cpu)
    mixer_data = DataLoader(
        mixer_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.n_cpu)

    # reals_label = torch.ones(C.batch_size)
    # fakes_label = torch.zeros(C.batch_size)

    for epoch in range(C.n_epoch):
        for n_iter, (swbd, mixer) in enumerate(zip(swbd_data, mixer_data)):
            swbd = swbd.unsqueeze(1)
            mixer = mixer.unsqueeze(1)

            swbd = swbd.cuda()
            mixer = mixer.cuda()
            # print(swbd.type(), mixer.type())
            # train generator
            loss_g, loss_gan_s2t, loss_gan_t2s, loss_idt_src, loss_idt_trg, loss_cyc_src, loss_cyc_trg = \
                generator_trian_step(
                    g_s2t, g_t2s, d_src, d_trg, mixer, swbd, gan_loss, identity_loss, cycle_loss, g_opt)
            # train source discriminator
            loss_d_src = discriminator_train_step(
                d_src, mixer, fake_src_buffer, gan_loss, d_src_opt)
            # train target discriminator
            loss_d_trg = discriminator_train_step(
                d_trg, swbd, fake_trg_buffer, gan_loss, d_trg_opt)

            if n_iter % C.report_interval == 0:
                print("[%4d/%4d] Iteration: %d" % (epoch, C.n_epoch, n_iter))
                print("G: %.6f, G_t2s: %.6f, G_s2t: %.6f" %
                      (loss_g, loss_gan_t2s, loss_gan_s2t))
                print("G_identity: %.6f, G_cycle: %.6f" % (
                    ((loss_idt_src+loss_idt_trg)*C.idt_lambda), C.cycle_gamma*(loss_cyc_src+loss_cyc_trg)))
                print("D: %.6f\n" % (loss_d_src+loss_d_trg))

        create_checkpoint(g_s2t, g_t2s, d_src, d_trg, epoch)
        validate_step()
