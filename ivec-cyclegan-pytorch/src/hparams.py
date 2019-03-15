from multiprocessing import cpu_count
import os.path as path
import os

SWBD_PATH="/gs/hs0/tga-tslab/baiyuu/data/ly/ivectors_swbd_train/"
MIXER_PATH="/gs/hs0/tga-tslab/baiyuu/data/ly/ivectors_mixer_train/"
EXP_DIR = "/home/3/17R17067/GitHub/LY/ivec-cyclegan-pytorch/exp"
TAG="test"
CKPT_PREFIX = "ckpt.tar.%s"
# generator setting
nc_input = 1
nc_output = 1
n_res_block = 9

# discriminator setting


# training
learning_rate = 0.0002
use_cuda = True
batch_size = 32

n_cpu = cpu_count()

n_epoch = 20

idt_lambda=5
cycle_gamma = 10

# other
report_interval = 10

# test setting
eval_condition="C5"
n_ckpt = "0"

EVAL_BASE_PATH="/gs/hs0/tga-tslab/baiyuu/data/ly/exp"
EVAL_FILES=[
    "ivectors_sre10_train",
    "ivectors_sre10_test",
    "ivectors_sre10_test_c5"
]
OUT_FILES=[
    "sre10_enroll.ark",
    "sre10_test.ark",
    "sre10_test_c5.ark"
]

evals=[ os.path.join(EVAL_BASE_PATH,eval_condition,files) for files in EVAL_FILES]

eval_outputs = [
    os.path.join(EXP_DIR,TAG,eval_condition,"ckpt_"+n_ckpt,"adapted_"+files) for files in EVAL_FILES
]