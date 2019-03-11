from multiprocessing import cpu_count

SWBD_PATH="/net/callisto/storage1/baiyuu/LY/data/swbd_train"
MIXER_PATH="/net/callisto/storage1/baiyuu/LY/data/mixer_train"

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