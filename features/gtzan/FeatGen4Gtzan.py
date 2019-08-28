import sys
sys.path.append('../fma_small/')

from FeatureGenerator import *
import h5py

DIR_AUDIO = 'D:/apsipa_2019/datasets/gtzan/'
DIR_FEATURE = 'D:/apsipa_2019/features/gtzan/'

class FeatGen4Gtzan(FeatureGenerator):
    def __init__(self, audio_dir, feature_dest):
        FeatureGenerator.__init__(self)
        self.audio_dir = audio_dir
        self.feature_dest = feature_dest
        self.dir_stft = feature_dest + \
            '/stft_s22khz_f1024_h512_db_small/'
        
        self.dir_mel_stft = feature_dest + \
            '/mel_stft_s22khz_f1024_h512_mel96_db_small/'
        
        self.dir_mel_harmonic = feature_dest + \
            '/mel_stft_harmonic_s22khz_f1024_h512_mel96_db_small/'
        
        self.dir_mel_percussive = feature_dest + \
            '/mel_stft_percussive_s22khz_f1024_h512_mel96_db_small/'
        
        self.dir_mfcc = feature_dest + \
            '/mfcc_d60_s22khz_f1024_h512_small/'
        
        self.genres = {'metal': 0, 'disco': 1, 'classical': 2,
                       'hiphop': 3, 'jazz': 4, 'country': 5,
                       'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
        
    def _save_files(self, feature_dir, genre, file, feature):
        # name the file
        feature_file = file[:-3] + ".npy"

        if not(os.path.exists(feature_dir +'/'+ genre)):
            os.makedirs(feature_dir +'/'+ genre)

        np.save(os.path.join(feature_dir +'/'+ genre, feature_file), 
                np.float32(feature))

    def gen_feat(self):
        bad_tracks_list = []
        ids = 1000
        count = 0

        for genre,_ in self.genres.items(): #!Sean!: watch the underscore..
            for root, subdirs, files in os.walk(self.audio_dir +'/'+ genre): 
                for count, file in enumerate(files):
                    # set file path
                    audio_path = os.path.join(root, file)
                    print(audio_path)
                    
                    features = self.gen_acoustic_features(audio_path)
                    dirs = [self.dir_stft, self.dir_mel_stft, 
                            self.dir_mel_harmonic, self.dir_mel_percussive,
                            self.dir_mfcc
                           ]
                    
                    # save files for 5 features
                    for i in range(5):
                        self._save_files(dirs[i], genre, file, features[i])
                    
                    count += 1
                    prog = count / ids * 100
                    if  prog % 5 == 0:
                        print('finish', prog, '%')
                        
    def _load_data(self, feature_dir, norm=False):
        TRACK_COUNT = 1000
        feature_shape = np.load(feature_dir + '/blues/blues.00000.npy').shape

        x = np.zeros(((TRACK_COUNT,) + feature_shape), dtype=np.float32)
        y = np.zeros((TRACK_COUNT, len(self.genres)), dtype=np.int8)

        for genre_name, genre_key in self.genres.items():
            for root, subdirs, files in os.walk(feature_dir + genre_name):
                for i, file in enumerate(files):
                    # determine the file path & loading
                    file_dir = feature_dir + genre_name + '/' + file
                    print(file_dir)
                    feature = np.load(file_dir)
                    if norm is True:
                        feature = (feature - np.mean(feature)) / np.std(feature)

                    # append the x data
                    idx_track = genre_key * 100 + i
                    print("No. ", idx_track, " songs")
                    x[idx_track] = feature

                    # append the y data
                    y[idx_track, genre_key] = 1 # append genre data
                    print(y[idx_track])

        print('feature data shape: ', x.shape)
        print(np.max(x), np.min(x), np.mean(x), np.std(x))
        return x, y
    
    def _load_data_ff(self, list_dir, feature_dir, norm=False):
        # get feature shape from sample data
        feat_shape = np.load(feature_dir + '/blues/blues.00000.npy').shape
    
        f = open(list_dir, 'r')
        file_list = f.read().split('\n')

        x = np.zeros((len(file_list),) + feat_shape, dtype=np.float32)
        y = np.zeros((len(file_list), len(self.genres)), dtype=np.int8)

        for i, file in enumerate(file_list):
            feature = np.load(feature_dir + file[:-3] + 'npy')
            if norm is True:
                feature = (feature - np.mean(feature)) / np.std(feature)
            x[i] = feature
            genre_key = self.genres[file_list[i].split('/')[0]]
            y[i, genre_key] = 1

        print('max, min, mean, std')
        print(np.max(x), np.min(x), np.mean(x), np.std(x))
        
        return x, y
    
    def package_hdf5(self):
        # for gtzan 10-fold cv
        feature_type = {'original' : self.dir_mel_stft, 
                        'harmonic' : self.dir_mel_harmonic,
                        'percussive' : self.dir_mel_percussive, 
                        'modulation' : self.dir_mel_stft,
                        'mfcc': self.dir_mfcc,
                       }
        
        for feat_name, feat_dir in feature_type.items():
            print('\nloading {}'.format(feat_name))
            
            x, y = self._load_data(feat_dir, norm=True)
            
            if feat_name is 'modulation':
                x = self._frequency_modulation(x, norm=True)
            
            f = h5py.File('./10fdcv/{}.hdf5'.format(feat_name),'w')
            f['x'] = x
            f['y'] = y
            f.close()
        
    def package_hdf5_ff(self):
        txt_train = \
        './gtzan_fault_filtered_split/train_filtered.txt'

        txt_valid = \
        './gtzan_fault_filtered_split/valid_filtered.txt'

        txt_test = \
        './gtzan_fault_filtered_split/test_filtered.txt'
        
        feature_type = {'original' : self.dir_mel_stft, 
                        'harmonic' : self.dir_mel_harmonic,
                        'percussive' : self.dir_mel_percussive, 
                        'modulation' : self.dir_mel_stft,
                        'mfcc': self.dir_mfcc,
                       }
        
        for feat_name, feat_dir in feature_type.items():
            print('\nloading {}'.format(feat_name))
            x_train, y_train = \
                self._load_data_ff(txt_train, feat_dir, norm=True)
            x_valid, y_valid = \
                self._load_data_ff(txt_valid, feat_dir, norm=True)
            x_test, y_test = \
                self._load_data_ff(txt_test, feat_dir, norm=True)
            
            if feat_name is 'modulation':
                x_train = self._frequency_modulation(x_train, norm=True)
                x_valid = self._frequency_modulation(x_valid, norm=True)
                x_test = self._frequency_modulation(x_test, norm=True)
            
            f = h5py.File('./h5/ff/{}.hdf5'.format(feat_name),'w')
            f['x_train'] = x_train
            f['x_valid'] = x_valid
            f['x_test'] = x_test
            f['y_train'] = y_train
            f['y_valid'] = y_valid
            f['y_test'] = y_test
            f.close()