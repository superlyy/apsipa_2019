from keras.models import Sequential, Model
from keras.layers import InputLayer, Lambda, Flatten, Dense
from keras.layers import Lambda, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import CuDNNGRU, Bidirectional, Input, Activation
from keras.layers import ELU
from keras.optimizers import Adam
from keras import backend as K

class Models:
    '''
    include expert model (CNN for ori, fm, mfcc)
    and mixer model (rnn), then mixture model (baseline, cnn, rnn...)
    '''
    def __init__(self, num_class):
        self.feature_types = ['original', 'harmonic', 'percussive', 'modulation']
        self.feat_input_shape = [(96, 1292), (96, 1292), (96, 1292), (96, 646)]
        self.num_class = num_class
        self.weight_file = './weights/{}.hdf5'
        
    def build_cnn_k2c2(self, feature_type, input_shape, num_class):
        # the cnn model has 5 conv layer in general
        # first we define the filter size in each layer
        filter_sizes = [64, 128, 128, 192, 256]
        kernel_size  = (3,3)
        
        if feature_type == 'original' or \
            feature_type == 'harmonic' or \
            feature_type == 'percussive' or \
            feature_type == 'mixer':
            pool_sizes = [(2,4), (2,4), (2,4), (3,5), (4,4)]
        elif feature_type == 'modulation':
            pool_sizes = [(2,2), (2,4), (2,5), (2,4), (4,4)]
        elif feature_type == 'mfcc':
            pool_sizes = [(2,4), (2,4), (2,5), (1,4), (2,4)]
        else:
            print('Please input valid token.\n')
            return 0
        
        model = Sequential(name='k2c2_{}'.format(feature_type))
        model.add(InputLayer(input_shape=input_shape+(1,),
                                name='input_{}'.format(feature_type)))

        for i in range(5):
            model.add(Conv2D(filters=filter_sizes[i], 
                                 kernel_size=kernel_size))
            model.add(BatchNormalization())
            model.add(ELU())
            model.add(MaxPooling2D(pool_size=pool_sizes[i], padding='same'))   

        model.add(Flatten())
        model.add(Dense(num_class, activation='softmax',
                                    name='dense_{}'.format(feature_type)))
        opt = Adam(lr=5e-3, decay=1e-2)
        model.compile(loss='categorical_crossentropy',  
                                   optimizer=opt,
                                    metrics=['accuracy'])  

        return model
    
    def build_rnn_mixer(self, input_shape, num_output):
        model_input = Input(input_shape, name='mixer_input')
        x = Dropout(0.25)(model_input)
        x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
        x = Dropout(0.25)(x)
        x = Bidirectional(CuDNNGRU(64, return_sequences=False))(x)
        x = Dropout(0.5)(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = Dense(num_output)(x)
        model_output = Activation('softmax', name='mixture_weights')(x)
        
        model = Model(inputs=model_input, outputs=model_output, name='mixer_rnn')
        opt = Adam(lr=5e-3, decay=1e-2)
        model.compile(loss='categorical_crossentropy',  optimizer=opt,
                  metrics=['accuracy'])  

        return model
    
    def _mix_func(self, x):
        # moe method
        weighted_sum = K.variable(0)
        mixer = x[-1]
        for i in range(len(x)-1):
            # multplied by its weighted (generated by mixer)
            # expand to 10 probabilities
            weighted_sum = weighted_sum + x[i] * K.tile(K.expand_dims(mixer[:,i],axis=-1),
                                [1,K.shape(x[i])[-1]])
        return weighted_sum

    def _mix_func_baseline(self, x):
        baseline_sum = K.variable(0)
        # balanced weight
        for model in x:
            baseline_sum = baseline_sum + model / len(x)

        return baseline_sum
    
    def _mix_output(self, input_shape):
        return input_shape[0]

    def _build_experts(self):
        self.experts = []
        for i in range(len(self.feature_types)):
            temp_model = self.build_cnn_k2c2(self.feature_types[i], 
                                            self.feat_input_shape[i], 
                                            self.num_class)
            temp_model.load_weights(self.weight_file.format(self.feature_types[i]))
            for layer in temp_model.layers:
                layer.trainable = False
                # layer.name = layer.name + '_' + str(self.feature_types[i])
            self.experts.append(temp_model)

    def build_moe_baseline(self):
        print('building MoEB')
        self._build_experts()
        moe_in = []
        moe_out = []
        for i in range(len(self.feature_types)):
            moe_in.append(self.experts[i].input)
            moe_out.append(self.experts[i].output)  

        mix = Lambda(self._mix_func_baseline, 
                     output_shape=self._mix_output)(moe_out)
        model = Model(inputs=moe_in, outputs=mix, name="moe_baseline")
        opt = Adam(lr=5e-4, decay=1e-4)
        model.compile(loss='categorical_crossentropy',  
                                       optimizer=opt,
                                        metrics=['accuracy'])  
        self.moeb = model
        
        return model
    
    def build_moe_mixer(self, mixer_type):
        print('building moe with {}'.format(mixer_type))
        self._build_experts()
        moe_in = []
        moe_out = []
        for i in range(len(self.feature_types)):
            moe_in.append(self.experts[i].input)
            moe_out.append(self.experts[i].output)

        if mixer_type == 'cnn':
            mixer = self.build_cnn_k2c2('mixer', 
                                        self.feat_input_shape[0],
                                        len(self.feature_types))
        elif mixer_type == 'rnn':
            mixer = self.build_rnn_mixer(self.feat_input_shape[0][::-1], 
                                            len(self.feature_types))
        else:
            print('Please input the mixer type in right format\n')
            return 0
        moe_in.append(mixer.input)
        moe_out.append(mixer.output)

        mix = Lambda(self._mix_func, output_shape=self._mix_output)(moe_out)
        model = Model(inputs=moe_in, outputs=mix, name='MoE')
        opt = Adam(lr=5e-4, decay=1e-4)
        model.compile(loss='categorical_crossentropy',  
                                       optimizer=opt,
                                        metrics=['accuracy'])

        return model


    ###---------------------------------------------------------------------
    ###temporary funcs for sean
    def export_expert_weights(self):
        self.build_moer()
        self.moer.load_weights('./weights/moer_559.hdf5')
        for feature_type in self.feature_types:
            temp_model = Model(inputs=self.moer.get_layer(
                                   'input_{}'.format(feature_type)).output,
                               outputs=self.moer.get_layer(
                                   'dense_{}'.format(feature_type)).output)
            temp_model.save_weights('./weights/{}_moer_559.hdf5'.format(feature_type))