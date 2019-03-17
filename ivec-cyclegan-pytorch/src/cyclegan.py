import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np

import itertools
from logger import *
import os
import random

from generator import Generator
from discriminator import Discriminator
import dataset
import hparams as C
from data_utils import ReplayBuffer
from model import CycleGAN
import evaluation



fake_src_buffer = ReplayBuffer()
fake_trg_buffer = ReplayBuffer()


def create_checkpoint(model, epoch):
    checkpoint_name = C.CKPT_PREFIX % str(epoch)
    PATH = os.path.join(C.EXP_DIR, C.TAG, checkpoint_name)
    # os.system("mkdir -p %s"%os.path.join(C.EXP_DIR,C.TAG))
    os.makedirs(os.path.join(C.EXP_DIR, C.TAG), exist_ok=True)
    model.save_checkpoint(PATH)
    # torch.save({
    #     "g_s2t": g_s2t.state_dict(),
    #     "g_t2s": g_t2s.state_dict(),
    #     "d_src": d_src.state_dict(),
    #     "d_trg": d_trg.state_dict(),
    # }, PATH)
    print("Save checkpoint on %s" % PATH)


def generator_trian_step(model, src_data, trg_data, gan_loss, identity_loss, cycle_loss, optim):
    model.train()

    reals_label = torch.ones(C.batch_size)
    if C.use_cuda:
        reals_label = reals_label.cuda()

    # identity loss
    identity_src = model.trg2src(src_data)
    loss_idt_src = identity_loss(identity_src, src_data)

    identity_trg = model.src2trg(trg_data)
    loss_idt_trg = identity_loss(identity_trg, trg_data)

    # gan loss
    fakes_trg = model.src2trg(src_data)
    preds_fakes_trg = model.discriminate_trg(fakes_trg)
    loss_gan_s2t = gan_loss(preds_fakes_trg, reals_label)

    fakes_src = model.trg2src(trg_data)
    preds_fakes_src = model.discriminate_src(fakes_src)
    loss_gan_t2s = gan_loss(preds_fakes_src, reals_label)

    fake_src_buffer.push(fakes_src)
    fake_trg_buffer.push(fakes_trg)

    # cycle loss
    recoverd_src = model.trg2src(fakes_trg)
    loss_cyc_src = cycle_loss(recoverd_src, src_data)

    recoverd_trg = model.src2trg(fakes_src)
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
    d.train()

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


def validate_step(model, best=None, validate_step="eer"):
    assert validate_on in list(
        best.keys()), "validation should be based on one of the ['eer','dcf2', 'dcf3']"
    scores = evaluation.main(model, in_folder_path=C.valid_files)
    if scores[list(best.keys()).index(validate_on)] < best[validate_on]:
        for i, key in enumerate(list(best.keys())):
            best[key] = scores[i]
        # os.symlink()


if __name__ == "__main__":
    
    random.seed(C.random_seed)
    np.random.seed(C.random_seed)
    torch.manual_seed(C.random_seed)
    torch.cuda.manual_seed(C.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    cycle_gan = CycleGAN(
        Generator(C.g_conv_ch,C.g_trans_ch,C.g_kernels, C.g_strides,C.g_n_res_block, C.g_leaky_slop),
        Generator(C.g_conv_ch,C.g_trans_ch,C.g_kernels, C.g_strides,C.g_n_res_block, C.g_leaky_slop),
        Discriminator(C.nc_input),
        Discriminator(C.nc_input)
    )

    logging.info(cycle_gan)
    if C.use_cuda:
        logging.info("Use GPU for traninig")
        cycle_gan.cuda()

    gan_loss = torch.nn.MSELoss()
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()

    g_opt = torch.optim.Adam(itertools.chain(cycle_gan.g_s2t.parameters(
    ), cycle_gan.g_t2s.parameters()), C.learning_rate, betas=(0.5, 0.999))
    d_src_opt = torch.optim.Adam(
        cycle_gan.d_src.parameters(), C.learning_rate, betas=(0.5, 0.999))
    d_trg_opt = torch.optim.Adam(
        cycle_gan.d_trg.parameters(), C.learning_rate, betas=(0.5, 0.999))

    best_scores = {
        "eer": float("inf"),
        "dcf2": float("inf"),
        "dcf3": float("inf")
    }

    # load data
    # source
    logging.info("Loading data from %s"%C.MIXER_PATH)
    mixer_dataset = dataset.IVecDataset(C.MIXER_PATH)
    logging.info("Loading data from %s"%C.SWBD_PATH)
    swbd_dataset = dataset.IVecDataset(C.SWBD_PATH)
    swbd_data = DataLoader(swbd_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.n_cpu)
    mixer_data = DataLoader(mixer_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.n_cpu)
    logging.info("Start training...")
    for epoch in range(1, C.n_epoch+1):
        for n_iter, (swbd, mixer) in enumerate(zip(swbd_data, mixer_data)):
            swbd = swbd.unsqueeze(1)
            mixer = mixer.unsqueeze(1)
            if C.use_cuda:
                swbd = swbd.cuda()
                mixer = mixer.cuda()

            # train generator
            loss_g, loss_gan_s2t, loss_gan_t2s, loss_idt_src, loss_idt_trg, loss_cyc_src, loss_cyc_trg = generator_trian_step(
                cycle_gan, mixer, swbd, gan_loss, identity_loss, cycle_loss, g_opt)
            # train source discriminator
            loss_d_src = discriminator_train_step(
                cycle_gan.d_src, mixer, fake_src_buffer, gan_loss, d_src_opt)
            # train target discriminator
            loss_d_trg = discriminator_train_step(
                cycle_gan.d_trg, swbd, fake_trg_buffer, gan_loss, d_trg_opt)

            if n_iter % C.report_interval == 0:
                logging.info("[%4d/%4d] Iteration: %d" % (epoch, C.n_epoch, n_iter))
                logging.info("G: %.6f, G_t2s: %.6f, G_s2t: %.6f" %(loss_g, loss_gan_t2s, loss_gan_s2t))
                logging.info("G_identity: %.6f, G_cycle: %.6f" % (((loss_idt_src+loss_idt_trg)*C.idt_lambda), C.cycle_gamma*(loss_cyc_src+loss_cyc_trg)))
                logging.info("D: %.6f" % (loss_d_src+loss_d_trg))

        create_checkpoint(cycle_gan, epoch)
        
        if epoch % C.ckpt_interval == 0:
            err, dcf2, dcf3 = evaluation.main(cycle_gan)
            rst_path = os.path.join(C.EXP_DIR, C.TAG, "result.txt")
            logging.info("Writing answer to %s"%rst_path)
            with open(rst_path, "a+") as f:
                f.write("Epoch %d: %f(EER), %f(DCF2), %f(DCF3)\n" % (epoch, err, dcf2, dcf3))
