from __future__ import print_function, division
import os
from torch.utils.data import Dataset
from .load_data import load_custom_dataset, load_site_info, load_dataset, load_site_info_custom_data



class Sample():
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.dosage = self.left+self.right

class RegionDataset(Dataset):
    def __init__(self, root_dir, index_region, region=1, chromosome='chr22', dataset=''):
        if dataset:
            number_of_regions = str(int(len(os.listdir(root_dir))/5))
            hap_dir = os.path.join(root_dir, f'{dataset}_{region}.hap.gz')
            legend_dir = os.path.join(root_dir, f'{dataset}_{region}.legend.gz')
            label_hap_dir = os.path.join(root_dir, f'{dataset}_{region}.gtruth.hap.gz')
            label_legend_dir = os.path.join(root_dir, f'{dataset}_{region}.gtruth.legend.gz')
            
            self.site_info_list = load_site_info_custom_data(label_legend_dir)
            self.haplotype_list, self.label_haplotype_list, self.a1_freq_list =\
                load_custom_dataset(hap_dir, legend_dir, label_hap_dir, label_legend_dir)

            print("[DATASET]:",self.haplotype_list.shape, self.label_haplotype_list.shape, self.a1_freq_list.shape)
            
        else:
            hap_dir = os.path.join(root_dir,f'{chromosome}_{region}.hap.gz')
            legend_dir = os.path.join(root_dir,f'{chromosome}_{region}.legend.gz')
            label_hap_dir = os.path.join(root_dir, f'{chromosome}_true.hap.gz')
            label_legend_dir = os.path.join(root_dir, f'{chromosome}_true.legend.gz') 
            panel_dir = os.path.join(root_dir, f'region_{region}.legend.gz')
            index_start = os.path.join(root_dir, f'{index_region}.txt')

            self.site_info_list = load_site_info(panel_dir)
            self.haplotype_list, self.label_haplotype_list, self.a1_freq_list =\
                load_dataset(hap_dir, legend_dir, label_hap_dir, label_legend_dir, self.site_info_list, index_start)
            print("[DATASET]:",self.haplotype_list.shape, self.label_haplotype_list.shape, self.a1_freq_list.shape)
            
        # self.list_input = [Sample(self.haplotype_list[i], self.haplotype_list[i+1]) for i in range(0, len(self.haplotype_list)/2, 2)]
        # self.list_label = [Sample(self.haplotype_list[i], self.haplotype_list[i+1]) for i in range(0, len(self.label_haplotype_list)/2, 2)]

    def __len__(self):
        return len(self.haplotype_list)
    
    def __getitem__(self, index):
        x = self.haplotype_list[index]
        y = self.label_haplotype_list[index]
        a1_freq_list = self.a1_freq_list
        return x, y, a1_freq_list

    

