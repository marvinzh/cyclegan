import torch
from torch import nn


class CycleGAN(nn.Module):
    def __init__(self, g_s2t, g_t2s, d_src, d_trg):
        super().__init__()
        self.g_s2t = g_s2t
        self.g_t2s = g_t2s
        self.d_src = d_src
        self.d_trg = d_trg

    def src2trg(self, src_data):
        return self.g_s2t(src_data)

    def trg2src(self, trg_data):
        return self.g_t2s(trg_data)

    def discriminate_src(self, src_data):
        return self.d_src(src_data)

    def discriminate_trg(self, trg_data):
        return self.d_trg(trg_data)

    def save_checkpoint(self, ckpt_path):
        torch.save({
            "g_s2t": self.g_s2t.state_dict(),
            "g_t2s": self.g_t2s.state_dict(),
            "d_src": self.d_src.state_dict(),
            "d_trg": self.d_trg.state_dict(),
        }, ckpt_path)

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.g_s2t.load_state_dict(ckpt["g_s2t"])
        self.g_t2s.load_state_dict(ckpt["g_t2s"])
        self.d_src.load_state_dict(ckpt["d_src"])
        self.d_trg.load_state_dict(ckpt["d_trg"])
