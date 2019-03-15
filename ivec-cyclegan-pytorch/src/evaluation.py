import torch
import os
import hparams as C
from generator import Generator
from discriminator import Discriminator
import data_utils

ARK2SCP_CMD = "copy-vector ark,t:%s ark,scp,t:%s,%s"


def load_checkpoint():
    PATH = os.path.join(C.EXP_DIR, C.TAG)
    ckpt_file = C.CKPT_PREFIX % str(C.n_ckpt)
    model_path = os.path.join(PATH, ckpt_file)
    print("load model at %s" % model_path)

    ckpt = torch.load(model_path)

    g_s2t = Generator(C.nc_input, C.nc_input, C.n_res_block)
    g_s2t.load_state_dict(ckpt["g_s2t"])

    return g_s2t

def adapt_ivec(model, data, labels, output_path):
    n_data = len(data)

    adapted=[]
    print("adapting")
    for i, (d, label) in enumerate(zip(data, labels)):
        d_tensor = torch.Tensor(d)
        d_tensor = d_tensor.view(1, 1, -1)
        adapted_d = model(d_tensor)
        adapted_d = adapted_d.detach().squeeze()
        adapted_d = list(adapted_d.numpy())
        adapted.append(adapted_d)
        print("[%5d/%d] Adapting i-vector" % (i, n_data))

    data_utils.adpt_ivec2kaldi(adapted, labels, arkfilepath=output_path)

def generate_and_run_sh(path, cmds):
    ARK2SCP_HEADER = '''
    #!/bin/bash
    . ./cmd.sh
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

def scoring():
    # return scores
    pass

def main(model, in_folder_path, out_file_path):
    cmds = []
    for in_file, out_file in zip(in_folder_path, out_file_path):
        print("reading file: %s" % in_file)
        out_path, _= os.path.split(out_file)
        data, labels = data_utils.datalist_load(in_file)
        os.makedirs(out_path, exist_ok=True)
        adapt_ivec(g_s2t, data, labels, out_file)
        cmds.append(ARK2SCP_CMD % (out_file, os.path.join(
            out_path, "ivector.ark"), os.path.join(out_path, "ivector.scp")))

    generate_and_run_sh("ivec_ark2scp.sh", cmds)
    # return scoring()
    # pass


if __name__ == "__main__":
    g_s2t = load_checkpoint()
    g_s2t.eval()
    main(g_s2t, C.eval_files, C.adapted_files)
    # cmds = []
    # for in_file, out_file in zip(C.eval_files, C.adapted_files):
    #     print("reading file: %s" % in_file)
    #     out_path, _= os.path.split(out_file)
    #     data, labels = data_utils.datalist_load(in_file)
    #     os.makedirs(out_path, exist_ok=True)
    #     adapt_ivec(g_s2t, data, labels, out_file)
    #     cmds.append(ARK2SCP_CMD % (out_file, os.path.join(
    #         out_path, "ivector.ark"), os.path.join(out_path, "ivector.scp")))

    # generate_and_run_sh("ivec_ark2scp.sh", cmds)
