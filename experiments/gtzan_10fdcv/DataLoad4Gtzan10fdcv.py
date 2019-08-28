import sys
sys.path.append('../fma_small/')
from DataLoader import *

class DataLoad4Gtzan10fdcv(DataLoader):
    def __init__(self):
        data_dir = '../../features/gtzan/h5/'
        DataLoader.__init__(self, data_dir=data_dir)
        
    # override
    def load_feature_data(self):
        if self.data_dir is None:
            print('data directory is unspecified')
            return 0

        else:
            if type(self.feature_types) is str:
                print('loading {} data'.format(feat_type))
                f = h5py.File(self.data_dir + 
                              '/{}.hdf5'.format(feat_type), 'r')
                x = np.expand_dims(f['x'][:], axis=-1)
                y = f['y'][:]
                f.close()
                return x, y
            
            elif type(self.feature_types) is list:
                print('we have multiple feature types:')
                x = []
                for feat_type in self.feature_types:
                    print('  loading {} data'.format(feat_type))
                    temp_data = []
                    f = h5py.File(self.data_dir + 
                                  '/{}.hdf5'.format(feat_type), 'r')
                    x.append(np.expand_dims(f['x'][:], axis=-1))
                    f.close()

            # load label
            f = h5py.File(self.data_dir + 
                          '/{}.hdf5'.format(self.feature_types[0]), 'r')
            y = f['y'][:]
            f.close()
            return x, y
    
    # override
    def load_moe_mixer_data(self, mixer_type):
        if type(self.feature_types) is str:
            print('this function is for multiple features input')
            return 0
        else: 
            if mixer_type == 'cnn':
                print('loading data for MoEC')
                x, y = self.load_feature_data()
                x.append(x[0])
                return x, y
            elif mixer_type == 'rnn':
                print('loading data for MoER')
                x, y = self.load_feature_data()
                x.append(np.transpose(np.squeeze(
                                x[0], axis=-1), (0,2,1)))
                return x, y
            else:
                print('Please input the right form')
                return 0