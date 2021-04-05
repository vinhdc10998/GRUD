#!/usr/bin/env python3


from contextlib import contextmanager
import gzip
import os
import subprocess
import numpy as np

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

def load_data(hap_file, legend_file, hap_true_file, legend_true_file, site_info_list):
    def one_hot(allele, a1_freq):
        if allele is None:
            return [1.0 - a1_freq, a1_freq]
        return [1 - allele, allele]

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
                # print(item, haplotype)
                allele = None
                if item != 'NA':
                    allele = int(item)
                    if swap_flag:
                        allele = 1 - allele
                # print(a1_freq)
                haplotype[marker_id] = one_hot(allele, a1_freq)
    for key in site_info_dict.keys():
        if key not in key_set:
            site_info = site_info_dict[key]
            marker_id = site_info.marker_id
            one_hot_value = one_hot(None, site_info.a1_freq)
            for haplotype in haplotype_list:
                haplotype[marker_id] = one_hot_value

    _, hap_lines = load_lines(hap_true_file)
    legend_header, legend_lines = load_lines(legend_true_file, header=True)
    legend_header = legend_header.split()
    legend_position_col = get_item_col(legend_header, 'position', legend_true_file)
    hap_lines_split = np.array([(line.split()) for line in hap_lines])
    del hap_lines
    legend_positions_list = np.array([int(line.split()[legend_position_col]) for line in legend_lines])
    legend_lines_split = np.array([(line.split()) for line in legend_lines])
    del legend_lines
    label_fw = []
    a1_freq_list = []
    for key in label_info_dict.keys():
        position = label_info_dict[key].position
        a1_freq = label_info_dict[key].a1_freq
        check_array = legend_positions_list == int(position)
        if np.sum(check_array) == 1:
            list_allel_one_hot = []
            for allele in hap_lines_split[check_array].astype(int)[0]:
                list_allel_one_hot.append(one_hot(allele, a1_freq)) 
            label_fw.append(list_allel_one_hot)
            a1_freq_list.append(a1_freq)
        else:
            a0, a1 = label_info_dict[key].a0, label_info_dict[key].a1
            ids = legend_lines_split[check_array][:, 0]
            for index, id in enumerate(ids):
                temp = id.split(":")
                if temp[2] == a0 and temp[3] == a1:
                    list_allel_one_hot = []
                    for allele in hap_lines_split[check_array].astype(int)[0]:
                        list_allel_one_hot.append(one_hot(allele, a1_freq))  
                    label_fw.append(list_allel_one_hot)
                    a1_freq_list.append(a1_freq)


    label_haplotype_list = np.array(label_fw, dtype=np.double).reshape(-1,label_site_count,2)
    haplotype_list = np.array(haplotype_list, dtype=np.double)
    a1_freq_list = np.array(a1_freq_list, dtype=np.double)
    return haplotype_list, label_haplotype_list, a1_freq_list
