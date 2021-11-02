from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from .load_data import load_custom_dataset, load_site_info, load_dataset, load_site_info_custom_data



class Sample():
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.dosage = self.left+self.right

class RegionDataset(Dataset):
    def __init__(self, root_dir, region=1, chromosome='chr22', dataset=''):
        if dataset:
            number_of_regions = str(int(len(os.listdir(root_dir))/5))
            hap_dir = os.path.join(root_dir, f'{dataset}_true_{str(region).zfill(len(number_of_regions))}.hap.gz')
            legend_dir = os.path.join(root_dir, f'{dataset}_true_{str(region).zfill(len(number_of_regions))}.legend.gz')
            label_hap_dir = os.path.join(root_dir, f'{dataset}_true_{str(region).zfill(len(number_of_regions))}_gtrue.hap.gz')
            label_legend_dir = os.path.join(root_dir, f'{dataset}_true_{str(region).zfill(len(number_of_regions))}_gtrue.legend.gz')
            
            self.site_info_list = load_site_info_custom_data(label_legend_dir)
            self.haplotype_list, self.label_haplotype_list, self.a1_freq_list =\
                load_custom_dataset(hap_dir, legend_dir, label_hap_dir, label_legend_dir)

            
        else:
            with open(os.path.join(root_dir, 'index.txt'), 'w+') as index_file:
                index_file.write("0")
            hap_dir = os.path.join(root_dir,f'{chromosome}_{region}.hap.gz')
            legend_dir = os.path.join(root_dir,f'{chromosome}_{region}.legend.gz')
            label_hap_dir = os.path.join(root_dir, f'{chromosome}_true.hap')
            label_legend_dir = os.path.join(root_dir, f'{chromosome}_true.legend.gz') 
            panel_dir = os.path.join(root_dir, f'region_{region}.legend.gz')
            index_start = os.path.join(root_dir, f'index.txt')

            self.site_info_list = load_site_info(panel_dir)
            self.haplotype_list, self.label_haplotype_list, self.a1_freq_list =\
                load_dataset(hap_dir, legend_dir, label_hap_dir, label_legend_dir, self.site_info_list, index_start)
        # self.list_input = torch.stack([torch.tensor([self.haplotype_list[i], self.haplotype_list[i+1]]) for i in range(0, len(self.haplotype_list), 2)])
        self.list_input = []
        for i in range(0, len(self.haplotype_list), 2):
            tmp  = torch.stack([self.haplotype_list[i], self.haplotype_list[i+1]])
            self.list_input.append(tmp)
        
        self.list_input = torch.stack(self.list_input)

        self.list_label = torch.stack([torch.stack([self.label_haplotype_list[i], self.label_haplotype_list[i+1]]) for i in range(0, len(self.label_haplotype_list), 2)])
        # print("[DATASET]:",self.list_input.shape, self.list_label.shape, self.a1_freq_list.shape)
        print("[DATASET]:",self.list_input.shape, self.list_label.shape, self.a1_freq_list.shape)

    def __len__(self):
        return len(self.list_input)
    
    def __getitem__(self, index):
        x = self.list_input[index]
        y = self.list_label[index]
        y_dosage = []
        a1_freq_list = self.a1_freq_list
        y_dosage= y[0] + y[1]

        # y = torch.tensor(y)
        return x, y, y_dosage, a1_freq_list

    

