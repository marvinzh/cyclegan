from multiprocessing import cpu_count

TAG="test"
SWBD_PATH="/gs/hs0/tga-tslab/baiyuu/data/ly/ivectors_swbd_train/"
MIXER_PATH="/gs/hs0/tga-tslab/baiyuu/data/ly/ivectors_mixer_train/"
EXP_DIR = "/home/3/17R17067/GitHub/LY/ivec-cyclegan-pytorch/exp"

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