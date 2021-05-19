import glob
import gzip
import json
import pickle
from os import path

"""
get all the files in a folder with determined suffix
"""


def get_all_files_in_folder(folder_path, suffix):
    file_paths = glob.glob(folder_path + suffix)
    return file_paths


"""
unzip a list of gz files to string lists 
"""


def unzip_gz_files(zip_file_list):
    unzipped_file_list = []
    for file in zip_file_list:
        f = gzip.open(file, 'rt')
        file_content = json.load(f)
        unzipped_file_list.append(file_content)
        f.close()
    return unzipped_file_list


"""
save data to the file "file_name"
"""


def save_data_to_file(data, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


"""
given a file_name, load it to a variable
"""


def load_file_to_data(file_name):
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    return data


"""
check if file_name exists
"""


def has_file(file_name):
    return path.exists(file_name)
