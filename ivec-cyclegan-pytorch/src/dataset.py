import torch
from torch.utils.data import Dataset
import data_utils

class SwbdMixer(Dataset):
    def __init__(self,swbd_path, mixer_path):
        super().__init__()
        swbd_data, swbd_labels = data_utils.datalist_load(swbd_path)
        mixer_data, mixer_labels = data_utils.datalist_load(mixer_path)
        self.swbd = swbd_data
        self.mixer = mixer_data
        

    def __getitem__(self,index):
        pass
    
    def __len__(self):
        pass