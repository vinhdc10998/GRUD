#!/usr/bin/env python3


from contextlib import contextmanager
import gzip
import os
import subprocess
import numpy as np
import sys
import linecache
import torch
import gc
def system(command):
    subprocess.call(command, shell=True)


def mkdir(dirname):
    if dirname.strip() != '':
        os.makedirs(dirname, exist_ok=True)


@contextmanager
def reading(filename):
    root, ext = os.path.splitext(filename)
    fp = gzip.open(filename, 'rt') if ext == '.gz' else open(filename, 'rt')
    try:
        yield fp
    finally:
        fp.close()


@contextmanager
def writing(filename):
    root, ext = os.path.splitext(filename)
    fp = gzip.open(filename, 'wt') if ext == '.gz' else open(filename, 'wt')
    try:
        yield fp
    finally:
        fp.close()


def get_item_col(header_items, item_name, filename=None):
    try:
        col = header_items.index(item_name)
    except ValueError:
        error_message = 'header \'{:s}\' not found'.format(item_name)
        if filename is not None:
            error_message += ' in ' + filename
        print(error_message, file=sys.stderr)
        sys.exit(0)
    return col

class SiteInfo:
    def __init__(self, id, position, a0, a1, a1_freq, array_marker_flag):
        self.id = id
        self.position = position
        self.a0 = a0
        self.a1 = a1
        self.a1_freq = a1_freq
        self.array_marker_flag = array_marker_flag

def load_site_info(legend_file):
    site_info_list = []
    with reading(legend_file) as fp:
        items = fp.readline().rstrip().split()
        id_col = get_item_col(items, 'id', legend_file)
        a0_col = get_item_col(items, 'a0', legend_file)
        a1_col = get_item_col(items, 'a1', legend_file)
        position_col = get_item_col(items, 'position', legend_file)
        marker_flag_col = get_item_col(
            items, 'array_marker_flag', legend_file)
        a1_freq_col = get_item_col(items, 'a1_freq', legend_file)
        for line in fp:
            items = line.rstrip().split()
            site_info = SiteInfo(
                items[id_col], items[position_col], items[a0_col],
                items[a1_col], float(items[a1_freq_col]),
                items[marker_flag_col] == '1')
            site_info_list.append(site_info)
    return site_info_list

def load_lines(filename,header=False):
    header_line = None
    lines = []
    with reading(filename) as fp:
        if header:
            header_line = fp.readline().rstrip()
        for line in fp:
            lines.append(line.rstrip())
            
    return header_line, np.array(lines)

def one_hot(allele, a1_freq):
    if allele is None:
        return [1. - a1_freq, a1_freq]
    return [1 - allele, allele]

def convert_maf(a1_freq):
    if a1_freq > 0.5:
        res = 1. - a1_freq
        if res == 0.:
            res = 0.00001
        return res
    return a1_freq

def load_dataset(hap_file, legend_file, hap_true_file, legend_true_file, site_info_list, index_start):
    site_info_dict = {}
    marker_site_count = 0
    label_site_count = 0
    label_info_dict = {}

    for site_info in site_info_list:
        if site_info.array_marker_flag:
            site_info.marker_id = marker_site_count
            key = '{:s} {:s} {:s}'.format(
                site_info.position, site_info.a0, site_info.a1)
            site_info_dict[key] = site_info
            marker_site_count += 1
        else:
            site_info.marker_id = label_site_count
            key = '{:s} {:s} {:s}'.format(
                site_info.position, site_info.a0, site_info.a1)
            label_info_dict[key] = site_info
            label_site_count += 1

    load_info_list = []
    key_set = set()
    with reading(legend_file) as fp:
        items = fp.readline().rstrip().split()
        a0_col = get_item_col(items, 'a0', legend_file)
        a1_col = get_item_col(items, 'a1', legend_file)
        position_col = get_item_col(items, 'position', legend_file)
        for line in fp:
            items = line.rstrip().split()
            position = items[position_col]
            a0 = items[a0_col]
            a1 = items[a1_col]
            swap_flag = False
            key = '{:s} {:s} {:s}'.format(position, a0, a1)
            if key not in site_info_dict:
                key = '{:s} {:s} {:s}'.format(position, a1, a0)
                if key not in site_info_dict:
                    load_info_list.append(None)
                    continue
                swap_flag = True
            key_set.add(key)
            site_info = site_info_dict[key]
            marker_id = site_info.marker_id
            a1_freq = site_info.a1_freq
            load_info_list.append([marker_id, swap_flag, a1_freq])
    sample_size = 0
    with reading(hap_file) as fp:
        items = fp.readline().rstrip().split()
        sample_size = len(items)
    haplotype_list = [[None] * marker_site_count for _ in range(sample_size)]
    with reading(hap_file) as fp:
        for i, line in enumerate(fp):
            load_info = load_info_list[i]
            if load_info is None:
                continue
            items = line.rstrip().split()
            marker_id, swap_flag, a1_freq = load_info
            for item, haplotype in zip(items, haplotype_list):
                allele = None
                if item != 'NA':
                    allele = int(item)
                    if swap_flag:
                        allele = 1 - allele
                haplotype[marker_id] = one_hot(allele, a1_freq)
    for key in site_info_dict.keys():
        if key not in key_set:
            site_info = site_info_dict[key]
            marker_id = site_info.marker_id
            one_hot_value = one_hot(None, site_info.a1_freq)
            for haplotype in haplotype_list:
                haplotype[marker_id] = one_hot_value
    positions = []
    for key in label_info_dict.keys():
        positions.append(label_info_dict[key].position)
    
    start_index = None
    with open(index_start, "r") as index_file:
        start_index = int(index_file.read())
    
    true_haplotype_list = []
    count = 0
    a1_freq_list = []
    imp_site_info_list = [
        site_info
        for site_info in site_info_list if not site_info.array_marker_flag
    ]
    flag = 0
    with reading(legend_true_file) as fp, reading(hap_true_file) as kp:
        header_line = fp.readline().rstrip().split()
        position_col = header_line.index("position")
        for i, (line_fp, line_kp) in enumerate(zip(fp, kp)):
            k = i+1
            if k < start_index:
                continue
            items = line_fp.rstrip().split()
            hap = line_kp.rstrip().split()
            position = items[position_col]
            if position in positions:
                index_position_info_dict = [i for i,val in enumerate(positions) if val==position]
                for t in index_position_info_dict:
                    if items[2] == imp_site_info_list[t].a0 and items[3] == imp_site_info_list[t].a1:   
                        if flag == 1:
                            flag = 0
                            continue   
                        if position == '17996285' and imp_site_info_list[t].a1 == 'ATCTC':
                            # print(items)
                            # print(position, imp_site_info_list[t].a0, imp_site_info_list[t].a1)
                            # print(hap)
                            flag = 1
                            # pause = input("PAUSE...")
                        true_haplotype_list.append(list(map(int, hap))) 
                        a1_freq_list.append(convert_maf(imp_site_info_list[t].a1_freq))
                        count += 1
            if count == label_site_count:
                with open(index_start, "w+") as index_file:
                    start_index = index_file.write(str(k))
                break
    true_haplotype_list = torch.tensor(true_haplotype_list).T
    haplotype_list = torch.tensor(haplotype_list)
    a1_freq_list = torch.tensor(a1_freq_list)
    return haplotype_list, true_haplotype_list, a1_freq_list

def load_custom_dataset(hap_dir, legend_dir, label_hap_dir, label_legend_dir):
    true_haplotype_list = []
    a1_freq_list = []
    haplotype_list = []
    with reading(hap_dir) as fp:
        for i, line in enumerate(fp):
            items = line.rstrip().split()
            tmp = []
            for allele in items:
                tmp.append(one_hot(int(allele), None))
            haplotype_list.append((list(map(list, tmp))))

    with reading(label_hap_dir) as fp: 
        for i, line in enumerate(fp):
            items = line.rstrip().split()            
            true_haplotype_list.append(list(map( int, items)))

    with reading(label_legend_dir) as fp:
        header = fp.readline().rstrip().split()
        af_col = header.index("af")
        for line in fp:
            items = line.rstrip().split()
            maf = convert_maf(float(items[af_col]))
            a1_freq_list.append(maf)
    haplotype_list = torch.transpose(torch.tensor(haplotype_list, dtype=torch.float),0,1)
    true_haplotype_list = torch.tensor(true_haplotype_list, dtype=torch.long).T
    a1_freq_list = torch.tensor(a1_freq_list, dtype=torch.float)

    return haplotype_list, true_haplotype_list, a1_freq_list


def load_site_info_custom_data(legend_file):
    site_info_list = []
    with reading(legend_file) as fp:
        items = fp.readline().rstrip().split()
        id_col = get_item_col(items, 'id', legend_file)
        a0_col = get_item_col(items, 'ref', legend_file)
        a1_col = get_item_col(items, 'alt', legend_file)
        position_col = get_item_col(items, 'position', legend_file)
        marker_flag_col = get_item_col(
            items, 'array_marker_flag', legend_file)
        a1_freq_col = get_item_col(items, 'af', legend_file)
        for line in fp:
            items = line.rstrip().split()
            site_info = SiteInfo(
                items[id_col], items[position_col], items[a0_col],
                items[a1_col], float(items[a1_freq_col]),
                items[marker_flag_col] == '0')
            site_info_list.append(site_info)
    return site_info_list
