import torch
from torch.utils.data import Dataset
import data_utils

class IVecDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data, self.label = data_utils.datalist_load(path)
        label = sorted(label)
        label_set = set(label)
        
        self.label2idx= {key:i for i,key in enumerate(label_set)}
        self.idx2label = {i:key for i, key in enumerate(label_set)}

    def __getitem__(self,index):
        return self.data[index], self.label2idx[self.label[index]]
    
    def __len__(self):
        assert len(self.data) == len(self.label)
        return len(self.data)
