import torch
import os
import hparams as C
from generator import Generator
from discriminator import Discriminator
from model import CycleGAN
import data_utils
from logger import *

ARK2SCP_CMD = "copy-vector ark,t:%s ark,scp,t:%s,%s"


def load_checkpoint():
    PATH = os.path.join(C.EXP_DIR, C.TAG)
    ckpt_file = C.CKPT_PREFIX % str(C.n_ckpt)
    model_path = os.path.join(PATH, ckpt_file)
    print("load model at %s" % model_path)
    cycle_gan = CycleGAN(
        Generator(C.g_conv_ch,C.g_trans_ch,C.g_kernels, C.g_strides,C.g_n_res_block, C.g_leaky_slop),
        Generator(C.g_conv_ch,C.g_trans_ch,C.g_kernels, C.g_strides,C.g_n_res_block, C.g_leaky_slop),
        Discriminator(C.nc_input),
        Discriminator(C.nc_input)
    )
    cycle_gan.load_checkpoint(model_path)
    if C.use_cuda:
        cycle_gan.cuda()

    return cycle_gan


def adapt_ivec(model, data, labels, output_path):
    logging.info("adapting i-vectors to %s" % output_path)
    if C.use_cuda:
        data = data.cuda()
        
    adapted = []
    for i, (d, label) in enumerate(zip(data, labels)):
        d_tensor = torch.Tensor(d)
        d_tensor = d_tensor.view(1, 1, -1)
        adapted_d = model.src2trg(d_tensor)
        adapted_d = adapted_d.detach().squeeze()
        adapted_d = list(adapted_d.cpu().numpy())
        adapted.append(adapted_d)

    data_utils.adpt_ivec2kaldi(adapted, labels, arkfilepath=output_path)


def generate_and_run_sh(path, cmds):
    ARK2SCP_HEADER = '''
    #!/bin/bash
    . ./path.sh
    set -e
    date
    echo "Create scp files for adapted ivectors in different epochs."
    '''
    with open(path, "w+") as f:
        f.write(ARK2SCP_HEADER)
        for cmd in cmds:
            f.write(cmd+"\n")

    os.chmod(path, mode=0o755)
    os.system("./%s" % path)


def get_score(score_path):
    with open(score_path) as f:
        data = f.readlines()

    eer, dcf2, dcf3 = data[4].strip().split()[1:]

    return float(eer), float(dcf2), float(dcf3)


def scoring():
    logging.info("Scoring...")
    cwd = os.getcwd()
    eval_path = os.path.join(C.EXP_DIR, C.TAG, C.eval_condition)
    plda_scr_path = os.path.join(C.PLDA_PATH, "exp/", C.eval_condition)
    if os.path.exists(plda_scr_path):
        os.system("rm -rf %s" % plda_scr_path)

    os.symlink(eval_path, plda_scr_path)

    os.chdir(C.PLDA_PATH)
    scr_path = os.path.join(C.EXP_DIR, C.TAG, "score")
    os.system("./run_plda.sh > %s" % scr_path)
    os.chdir(cwd)
    eer, dcf2, dcf3 = get_score(scr_path)
    logging.info("score: %f, %f, %f" % (eer, dcf2, dcf3))
    return eer, dcf2, dcf3


def main(model, in_folder_path=C.test_files, out_file_path=C.adapted_files):
    cmds = []
    model.eval()
    for in_file, out_file in zip(in_folder_path, out_file_path):
        logging.info("reading file: %s" % in_file)
        out_path, _ = os.path.split(out_file)
        data, labels = data_utils.datalist_load(in_file)
        os.makedirs(out_path, exist_ok=True)
        data = torch.Tensor(data)
        adapt_ivec(model, data, labels, out_file)
        cmds.append(ARK2SCP_CMD % (out_file, os.path.join(
            out_path, "ivector.ark"), os.path.join(out_path, "ivector.scp")))

    generate_and_run_sh("ivec_ark2scp.sh", cmds)
    os.remove("ivec_ark2scp.sh")
    return scoring()
    # pass


if __name__ == "__main__":
    model = load_checkpoint()
    main(model, C.test_files, C.adapted_files)
