import numpy as np
import h5py

# capable for 
class DataLoader():
    def __init__(self, data_dir=None, specific_type=None):
        self.data_dir = data_dir
        if specific_type != None:
            # such as 'mfcc'
            self.feature_types = specific_type
        else:
            self.feature_types = ['original', 'harmonic', 
                                  'percussive', 'modulation']
    
    def load_feature_data(self):
        if self.data_dir is None:
            print('data directory is unspecified')
            return 0
        else:
            if type(self.feature_types) is str:
                print('loading {} data'.format(self.feature_types))
                x = []
                y = []
                f = h5py.File(self.data_dir + 
                              '/{}.hdf5'.format(self.feature_types), 'r')
                x.append(np.expand_dims(f['x_train'][:], axis=-1))
                x.append(np.expand_dims(f['x_valid'][:], axis=-1))
                x.append(np.expand_dims(f['x_test'][:], axis=-1))
                y.append(f['y_train'][:])
                y.append(f['y_valid'][:])
                y.append(f['y_test'][:])
                f.close()
                return x, y
            
            elif type(self.feature_types) is list:
                print('we have multiple feature types:')
                x_train = []
                x_valid = []
                x_test = []
                y = []
                
                for feat_type in self.feature_types:
                    print('  loading {} data'.format(feat_type))
                    temp_data = []
                    f = h5py.File(self.data_dir + 
                                  '/{}.hdf5'.format(feat_type), 'r')
                    x_train.append(np.expand_dims(f['x_train'][:], axis=-1))
                    x_valid.append(np.expand_dims(f['x_valid'][:], axis=-1))
                    x_test.append(np.expand_dims(f['x_test'][:], axis=-1))
                    f.close()

            # load label
            f = h5py.File(self.data_dir + 
                          '/{}.hdf5'.format(self.feature_types[0]), 'r')
            y.append(f['y_train'][:])
            y.append(f['y_valid'][:])
            y.append(f['y_test'][:])
            f.close()
            return [x_train, x_valid, x_test], y
        
    def load_moe_mixer_data(self, mixer_type):
        if type(self.feature_types) is str:
            print('this function is for multiple features input')
            return 0
        else: 
            if mixer_type == 'cnn':
                print('loading data for MoEC')
                x, y = self.load_feature_data()
                # 0: train, 1: valid, 2: test
                x[0].append(x[0][0])
                x[1].append(x[1][0])
                x[2].append(x[2][0])
                return x, y
            elif mixer_type == 'rnn':
                print('loading data for MoER')
                x, y = self.load_feature_data()
                x[0].append(np.transpose(np.squeeze(
                                x[0][0], axis=-1), (0,2,1)))
                x[1].append(np.transpose(np.squeeze(
                                x[1][0], axis=-1), (0,2,1)))
                x[2].append(np.transpose(np.squeeze(
                                x[2][0], axis=-1), (0,2,1)))
                return x, y
            else:
                print('Please input the right form')
                return 0