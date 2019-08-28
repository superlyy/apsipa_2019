import sys
sys.path.append('../fma_small/')
from Models import *

class Models4Gtzan10fdcv(Models):
    def __init__(self, n_fold):
        num_class = 10
        Models.__init__(self, num_class)
        self.weight_file = './weights/{}_fold_' + str(n_fold) + '.hdf5'