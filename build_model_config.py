import os
import numpy as np
import json
from data import utils

#TODO:
#build file config ( tam thoi dung` san cua model cung cap)
class ModelConfig():
    def __init__(self, hidden_units, num_layers, num_outputs, feature_size, input_dim, num_input, scope, output_points_fw, output_points_bw, num_classes):
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.feature_size = feature_size
        self.input_dim = input_dim
        self.num_input = num_input
        self.scope = scope #Higher or Lower model
        self.output_points_fw = output_points_fw
        self.output_points_bw = output_points_bw
        self.num_classes = 2

def write_config_model(region_file):
    output_points_fw = []
    output_points_bw = []
    with utils.reading(region_file) as rf:
        items = rf.readline().rstrip().split()
        position_col = utils.get_item_col(items, 'position', region_file)
        array_marker_flag = utils.get_item_col(items, 'array_marker_flag', region_file)
        print(position_col, array_marker_flag)
        i = 2
        for index, line in enumerate(rf):
            items = line.rstrip().split()
            if items[array_marker_flag] == '1':
                if index == 0:
                    pass
            elif items[array_marker_flag] == '0':
                pass
            if i==10:
                break
            i+=1
        # for line in fp:
    pass

def main():
    data_dir = 'data/org_data/region_1.legend.gz'
    write_config_model(data_dir)

if __name__ == "__main__":
    main()
