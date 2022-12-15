"""CCB9 images classification dataset.
   support:python3.x
"""
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import pickle

def load_data():
    """Loads CCB9 dataset.

    # Returns
        Tuple of Numpy arrays
    """
    def download(url):
        urlretrieve(url, url.split("/")[-1])

    def load_file(fpath):
        with open(fpath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        rgb = [data[b'data'][x:x+3,:] for x in range(0, len(data[b'data']), 3)]
        x = [rgb[x].transpose(1, 0).reshape([32, 32, 3]) for x in range(len(rgb))]
        y = data[b'labels']
        return x, y

    train_file_names = ['data_batch_%d'%(i) for i in range(1,5)]
    test_file_names = ['test_batch']

    if not os.path.exists('cucumber-9-python.tar.gz'):
        download('https://github.com/workpiles/CUCUMBER-9/'\
                 'raw/master/prototype_1/cucumber-9-python.tar.gz')

    with tarfile.open('cucumber-9-python.tar.gz', 'r:gz') as tf:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf)

    x_train = []
    y_train = []
    for fname in train_file_names:
        x, y = load_file(fname)
        x_train.extend(x)
        y_train.extend(y)

    x_test = []
    y_test = []
    for fname in test_file_names:
        x, y = load_file(fname)
        x_test.extend(x)
        y_test.extend(y)

    x_train = np.array(x_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.uint8)
    x_test = np.array(x_test, dtype=np.uint8)
    y_test = np.array(y_test, dtype=np.uint8)

    return x_train, y_train, x_test, y_test



if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
