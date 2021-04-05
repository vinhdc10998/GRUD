import os
import utils
import numpy as np

def load_lines(filename, header=False):
    header_line = None
    lines = []
    with utils.reading(filename) as fp:
        line_count = 0
        if header:
            header_line = fp.readline().rstrip()
        for line in fp:
            lines.append(line.rstrip())
    return header_line, lines

def true2label(hap_true_dir, legend_true_dir, region_dir):
    _, hap_lines = load_lines(hap_true_dir)
    legend_header, legend_lines = load_lines(legend_true_dir, True)
    region_header, region_lines = load_lines(region_dir, True)
    print("[Legend Header]: ", legend_header)
    print("[Region Header]: ", region_header)

    legend_header, region_header = legend_header.split(), region_header.split()

    legend_position_col = utils.get_item_col(legend_header, 'position', legend_true_dir)
    region_position_col = utils.get_item_col(region_header, 'position', region_dir)
    region_flag_col = utils.get_item_col(region_header, 'array_marker_flag', region_dir)

    legend_positions_list = np.array([int(line.split()[legend_position_col]) for line in legend_lines])
    hap_lines_split = np.array([line.split() for line in hap_lines])

    hap_labels_fw,  hap_labels_bw= [], []
    label_fw, label_bw = [], []
    legend_lines_split = np.array([(line.split()) for line in legend_lines])

    for line in region_lines:
        items = line.rstrip().split()
        position = int(items[region_position_col])
        if int(items[region_flag_col]) == 1:
            hap_labels_fw.append(label_fw)
            label_fw = []
        else:
            check_array = legend_positions_list == position
            if np.sum(check_array) == 1:
                label_fw.append([hap_lines_split[check_array].astype(int)[0]])
            else:
                id_region = items[0].split(":")
                ids = legend_lines_split[check_array][:, 0]
                for index, id in enumerate(ids):
                    temp_id = id.split(":")
                    if temp_id[2] == items[2] and temp_id[3] == items[3]:
                        print(index, id, items[2], items[3], temp_id[2], temp_id[3])
                        label_fw.append([hap_lines_split[check_array].astype(int)[index]])


    hap_labels_fw = np.asarray(hap_labels_fw)
    print(hap_labels_fw[:10])
    
def main():
    hap_true_dir = 'data/org_data/chr22_true.hap.gz'
    legend_true_dir = 'data/org_data/chr22_true.legend.gz'
    region_dir = 'data/org_data/region_1.legend.gz'

    true2label(hap_true_dir, legend_true_dir, region_dir)

if __name__ == "__main__":
    main()