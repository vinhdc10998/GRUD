from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from .load_data import load_dataset_inference, load_site_info

class RegionInferenceDataset(Dataset):
    def __init__(self, root_dir, region=1, chromosome='chr22'):
        hap_dir = os.path.join(root_dir,f'{chromosome}_{region}.hap.gz')
        legend_dir = os.path.join(root_dir,f'{chromosome}_{region}.legend.gz')
        panel_dir = os.path.join(root_dir, f'region_{region}.legend.gz')

        self.site_info_list = load_site_info(panel_dir)
        self.haplotype_list = load_dataset_inference(hap_dir, legend_dir, self.site_info_list)
        print("[DATASET]:",self.haplotype_list.shape)

    def __len__(self):
        return len(self.haplotype_list)
    
    def __getitem__(self, index):
        x = self.haplotype_list[index]
        return x
    

