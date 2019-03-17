from multiprocessing import cpu_count
import os.path as path
import os

SWBD_PATH = "/gs/hs0/tga-tslab/baiyuu/data/ly/ivectors_swbd_train/"
MIXER_PATH = "/gs/hs0/tga-tslab/baiyuu/data/ly/ivectors_mixer_train/"
EXP_DIR = "/home/3/17R17067/GitHub/LY/ivec-cyclegan-pytorch/exp"
TAG = "test"
CKPT_PREFIX = "ckpt.tar.%s"

# generator setting
random_seed = 0
nc_input = 1
nc_output = 1
g_n_res_block = 9
g_conv_ch = [nc_input, 32, 64, 128]
g_trans_ch = [128, 64, 32]
g_kernels = [3, 3, 3, -1, 3, 3, 3]
g_strides = [1, 2, 2, -1, 2, 2, 1]
g_leaky_slop = 0.2

# discriminator setting


# training
learning_rate = 0.0002
use_cuda = True
batch_size = 32

n_cpu = cpu_count()

n_epoch = 40

idt_lambda = 5
cycle_gamma = 10

# other
report_interval = 100
ckpt_interval = 5

# test setting
eval_condition = "C5"
n_ckpt = "1"

EVAL_BASE = "/gs/hs0/tga-tslab/baiyuu/data/ly/exp"
TEST_FOLDER = [
    "ivectors_sre10_train",
    "ivectors_sre10_test",
    #    "ivectors_sre10_test_c5"
]

test_files = [os.path.join(EVAL_BASE, eval_condition, files)
              for files in TEST_FOLDER]

VALID_FOLDER = [
    "ivectors_sre10_train",
    "ivectors_sre10_dev",
]

valid_files = [os.path.join(EVAL_BASE, eval_condition, files)
               for files in VALID_FOLDER]

adapted_files = [
    os.path.join(EXP_DIR, TAG, eval_condition,
                 "adapted_train", "sre10_enroll.ark"),
    os.path.join(EXP_DIR, TAG, eval_condition,
                 "adapted_test", "sre10_test.ark"),
    #   os.path.join(EXP_DIR,TAG,eval_condition,"adapted_test_c5","sre10_test_c5.ark")
]

# scoring
PLDA_PATH = "/home/3/17R17067/GitHub/LY/ivec-cyclegan-pytorch/src/plda_scoring/"

# adapted_eval_folder = [
#     os.path.join(EXP_DIR, TAG, eval_condition, "adapted_"+files) for files in TEST_FOLDER
# ]
