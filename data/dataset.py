from __future__ import print_function, division

import os
import torch
from torch.utils.data import Dataset
from .load_data import load_site_info, load_dataset


class RegionDataset(Dataset):
    def __init__(self, root_dir, region=1, chromosome='chr22'):
        hap_dir = os.path.join(root_dir,f'{chromosome}_{region}.hap.gz')
        legend_dir = os.path.join(root_dir,f'{chromosome}_{region}.legend.gz')
        label_hap_dir = os.path.join(root_dir, f'{chromosome}_true.hap')
        label_legend_dir = os.path.join(root_dir, f'{chromosome}_true.legend.gz') 
        panel_dir = os.path.join(root_dir, f'region_{region}.legend.gz')
        index_start = os.path.join(root_dir, f'index.txt')

        self.site_info_list = load_site_info(panel_dir)
        self.haplotype_list, self.label_haplotype_list, self.a1_freq_list =\
            load_dataset(hap_dir, legend_dir, label_hap_dir, label_legend_dir, self.site_info_list, index_start)
        print("[DATASET]:",self.haplotype_list.shape, self.label_haplotype_list.shape, self.a1_freq_list.shape)
    
    def __len__(self):
        return len(self.haplotype_list)
    
    def __getitem__(self, index):
        x = self.haplotype_list[index]
        y = self.label_haplotype_list[index]
        a1_freq_list = self.a1_freq_list
        return x, y, a1_freq_list
