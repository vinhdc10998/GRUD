from __future__ import print_function, division
import os
import numpy as np
from .utils import load_site_info, load_data
from torch.utils.data import Dataset

class RegionDataset(Dataset):
    def __init__(self, root_dir, region=1, chromosome='chr22'):
        hap_dir = os.path.join(root_dir,f'{chromosome}_{region}.hap.gz')
        legend_dir = os.path.join(root_dir,f'{chromosome}_{region}.legend.gz')
        label_hap_dir = os.path.join(root_dir, f'{chromosome}_true.hap.gz')
        label_legend_dir = os.path.join(root_dir, f'{chromosome}_true.legend.gz') 
        panel_dir = os.path.join(root_dir, f'region_{region}.legend.gz')

        self.site_info_list = load_site_info(panel_dir)
        self.haplotype_list, self.label_haplotype_list, self.a1_freq_list =\
            load_data(hap_dir, legend_dir, label_hap_dir, label_legend_dir, self.site_info_list)
        print("[DATASET]:",self.haplotype_list.shape, self.label_haplotype_list.shape, self.a1_freq_list.shape)
    
    def __len__(self):
        return len(self.haplotype_list)
    
    def __getitem__(self, index):
        return np.array(self.haplotype_list[index], dtype=np.double),\
            np.array(self.label_haplotype_list[index],dtype=np.double)
