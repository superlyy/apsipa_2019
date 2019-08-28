from FeatureGenerator import *
import h5py

PATH_TRACK_INFO = 'D:/fma/dataset/fma_metadata/tracks.csv'
DIR_AUDIO = 'D:/apsipa_2019/datasets/fma_medium/'
DIR_FEATURE = 'D:/apsipa_2019/features/fma_small/'

class FeatGen4FmaSmall(FeatureGenerator):
    
    def __init__(self, track_info_path, audio_dir, feature_dest):
        FeatureGenerator.__init__(self)
        # get info of tracks.csv
        self.tracks = self._load_track_info(track_info_path)
        self.audio_dir = audio_dir
        self.feature_dest = feature_dest
        self.dir_stft = feature_dest + \
            '/stft_s22khz_f1024_h512_db_small'
        
        self.dir_mel_stft = feature_dest + \
            '/mel_stft_s22khz_f1024_h512_mel96_db_small'
        
        self.dir_mel_harmonic = feature_dest + \
            '/mel_stft_harmonic_s22khz_f1024_h512_mel96_db_small'
        
        self.dir_mel_percussive = feature_dest + \
            '/mel_stft_percussive_s22khz_f1024_h512_mel96_db_small'
        
        self.dir_mfcc = feature_dest + \
            '/mfcc_d60_s22khz_f1024_h512_small'
    
    def _load_track_info(self, filepath):
        # the input path should be tracks.csv from fma doc
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]

        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            pd.api.types.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'license'), ('artist', 'bio'),


                   ('album', 'type'), ('album', 'information')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')
        
        # loc as fma small
        subset = tracks.index[tracks['set', 'subset'] == 'small']
        tracks = tracks.loc[subset]
        assert subset.isin(tracks.index).all()
        
        self.track_ids_train = tracks.index[tracks['set', 'split'] == 'training']
        self.track_ids_valid = tracks.index[tracks['set', 'split'] == 'validation']
        self.track_ids_test = tracks.index[tracks['set', 'split'] == 'test']

        return tracks
    
    def gen_feat(self):
        self.bad_tracks_list = []

        ids = self.tracks.index
        count = 0
        for track_id in ids:
            try:            
                wav_path = self._get_audio_path(self.audio_dir, track_id)

                features = self.gen_acoustic_features(audio_path)
                    dirs = [self.dir_stft, self.dir_mel_stft, 
                            self.dir_mel_harmonic, self.dir_mel_percussive,
                            self.dir_mfcc
                           ]

                # save files for 5 features
                    for i in range(5):
                        self._save_files(track_id, dirs[i], features[i])

                count += 1
                prog = count / len(ids) * 100
                if  prog % 5 == 0:
                    print('finish', prog, '%')
            except:
                print('id {0} is cracked'.format(track_id))
                self.bad_tracks_list.append(track_id)

        print('finsih extracting')
    
    def _get_audio_path(self, audio_dir, track_id):
        """
        Return the path to the mp3 given the directory where the audio is stored
        and the track ID.

        Examples
        --------
        >>> import utils
        >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
        >>> utils.get_audio_path(AUDIO_DIR, 2)
        '../data/fma_small/000/000002.mp3'

        """
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')
    
    def _get_feature_path(self, feature_dir, track_id):
        tid_str = '{:06d}'.format(track_id)
        return os.path.join(feature_dir, tid_str[:3], tid_str + '.npy')
    
    def _save_feature(self, track_id, feature_dir, feature):
        # create filename
        tid_str = '{:06d}'.format(track_id)
        feature_file = tid_str + '.npy'
        subdir = feature_dir + '/' + tid_str[:3]

        # saving feature
        if not(os.path.exists(subdir)):
            os.makedirs(subdir)

        np.save(os.path.join(subdir, feature_file), 
                np.float32(feature))
        
    def _load_data(self, x, feature_dir, track_id_list, norm=False):
        # x should be (data_size, feature_size)
        start = time.time()
        for i, track_id in enumerate(track_id_list):
            feature = np.load(self._get_feature_path(feature_dir, track_id))
            if norm is False:
                x[i] = feature
            else:
                x[i] = (feature - np.mean(feature)) / np.std(feature)
        end = time.time()
        print('time cost of loading data: ', end-start)    
        print('shape of data: ', x.shape)
        
    def create_label(self):
        tracks_genre = self.tracks['track', 'genre_top']
        genres = list(LabelEncoder().fit(tracks_genre).classes_)
                      
        print('Genres in FMA small ({}): {}'.format(len(genres), genres))        
        print(tracks_genre.value_counts(normalize=False))

        y = np.zeros((len(tracks_genre), len(genres)), dtype=np.int8)
        for i, track_genre in enumerate(tracks_genre):
            idx_onehot = genres.index(track_genre)
            y[i, idx_onehot] = 1
        print(y.shape)

        y = pd.DataFrame(y, index=self.tracks.index)

        self.y_train = y.loc[self.track_ids_train].values
        self.y_valid = y.loc[self.track_ids_valid].values
        self.y_test = y.loc[self.track_ids_test].values
        
    def package_to_hdf5(self):
        '''
        we package features in hdf5 format
        
        we should run function of label creating first
        '''
        self.create_label()
        
        feature_type = {'original' : self.dir_mel_stft, 
                        'harmonic' : self.dir_mel_harmonic,
                        'percussive' : self.dir_mel_percussive, 
                        'modulation' : self.dir_mel_stft,
                        'mfcc': self.dir_mfcc,
                       }
        
        for feat_name, feat_dir in feature_type.items():
            print('loding {}'.format(feat_name))
            # get feature shape from sample data
            feat_shape = np.load(self._get_feature_path(feat_dir, 2)).shape
            print(feat_shape)

            ## load train data
            x_train = np.zeros((self.track_ids_train.shape[0],) + feat_shape, 
                               dtype=np.float32)
            self._load_data(x_train, feat_dir, self.track_ids_train, norm=True)

            # load validation data
            x_valid = np.zeros((self.track_ids_valid.shape[0],) + feat_shape, 
                               dtype=np.float32)
            self._load_data(x_valid, feat_dir, self.track_ids_valid, norm=True)

            # load test data
            x_test = np.zeros((self.track_ids_test.shape[0],) + feat_shape, 
                              dtype=np.float32)
            self._load_data(x_test, feat_dir, self.track_ids_test, norm=True)
            
            if feat_name is 'modulation':
                x_train = self._frequency_modulation(x_train, norm=True)
                x_valid = self._frequency_modulation(x_valid, norm=True)
                x_test = self._frequency_modulation(x_test, norm=True)
            
            f = h5py.File('./test_func/{}.hdf5'.format(feat_name), 'w')
            f['x_train'] = x_train
            f['x_valid'] = x_valid
            f['x_test'] = x_test
            f['y_train'] = self.y_train
            f['y_valid'] = self.y_valid
            f['y_test'] = self.y_test
            f.close()
            print('finish saving {}'.format(feat_name))