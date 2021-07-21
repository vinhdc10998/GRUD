import os

for config_file in os.listdir('.'):
    # print(config_file)
    if 'config2' in config_file:
        os.remove(config_file)
    if 'config1' in config_file:
        file_name = config_file.replace('config1', 'config')
        # print(file_name)
        os.rename(config_file, file_name)