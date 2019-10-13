import os, random
import numpy as np
import tensorflow.keras as keras
import librosa



class AudioDataGenerator(keras.utils.Sequence):
    """ Generates data for Keras """
    
       
    def __init__(self, bg_noise_path, list_IDs, all_classes, base_path, dim_2, dim_spec=(20, 44), labels=[''],
                 batch_size=32,shuffle=True, n_channels=1, sr=16000, n_fft=2048, n_mels=44,
                 hop_length=512, power=2.0, ref_log_scal=1, n_mfcc=20, option='mel'):
        """Initialization"""
        
        self.base_path = base_path
        self.bg_noise_path = bg_noise_path
        self.list_noises = os.listdir(bg_noise_path)
        self.batch_size = batch_size
        self.all_classes = all_classes
        self.dim = (batch_size, *dim_spec)
        self.dim_2 = dim_2
        self.option = option
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.ref_log_scal = ref_log_scal
        self.labels = labels
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.list_IDs = list_IDs
        self.n = 0
        self.max = self.__len__()
        self.on_epoch_end()
    
    def __len__(self):
        """ Denotes the number of batches per epoch """
        
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    
     
        
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
                   
                   
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def wave2mfcc(self, path_wav):
        
        # Open randomly a background-noise
        choice = random.choice(self.list_noises)
        
        noise, srate = librosa.load(self.bg_noise_path+'\\'+choice, duration=1, mono=True, sr=16000)
        if len(noise) < 16000:
            noise = np.array(np.pad(noise, (0, self.dim_2[1] - len(noise)), 'constant', constant_values= 0))
        else:
            noise = noise[:16000]
      
        # Open the audio file
        wave, srate = librosa.load(path_wav, duration=1, mono=True, sr=16000)
        
        if len(wave) < 16000:
            wave = np.array(np.pad(wave, (0, 16000 - len(wave)), 'constant', constant_values= 0))
        else:
            wave = wave[:16000]
        # Mixing audio with noise
        alpha = random.choice(list(np.arange(0,0.25,0.02)))
        wave = alpha * noise + (1 - alpha) * wave
                
                
        # We create the mfcc
        mfccs = librosa.feature.mfcc(y=wave, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                                    power=self.power, n_mels=self.dim[1])
        
        wave = librosa.resample(wave, srate, self.dim_2[1])

        
        return wave, mfccs
    
    def wave2mspec(self, path_wav):
        wave, srate = librosa.load(path_wav, duration=2, mono=True, sr=None)
        
        # We create the mel spectrogramm
        mspec = librosa.feature.melspectrogram(y=wave, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                                    power=self.power, n_mels=self.n_mels)
        # Then we will convert into decibel(dB) because it is a good practice
        mspec = librosa.amplitude_to_db(mspec, ref=self.ref_log_scal)
        
        wave = librosa.resample(wave, srate, self.sr)
        
        return wave, mspec    
            
            
    def __data_generation(self, list_IDs_temp):
        """ Generates data containing batch_size samples """
                   
       # Initialization 
        #X_1 = np.empty((*self.dim, self.n_channels))
        X_1 = np.empty((self.dim[0], self.dim[1], self.n_mels, self.n_channels))
        X_2 = np.empty((self.batch_size, *self.dim_2))
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            cla = self.labels[ID]
            path = os.path.join(self.base_path, cla)
            path = os.path.join(path, ID)
 
            # Store sample
            #audio, sr = librosa.load(path, offset=0, duration=2)

            #mat = librosa.feature.mfcc(y=audio, sr=sr)
            if self.option == 'mel':
                mat = self.wave2mspec(path_wav=path)
                mat = mat[1]
                audio = mat[0]
            else:
                mat = self.wave2mfcc(path_wav=path)
                mat = mat[1]
                audio = mat[0]
            
            if mat.shape[1] < self.dim[2]:
                mat = np.array(np.pad(mat, ((0,0), (0, self.n_mels - mat.shape[1])),'constant', constant_values= 0))
              
            mat = mat.reshape((*mat.shape,1))
            
            X_1[i,] = mat
            
            #audio = librosa.resample(audio, sr, 8000)
           
            if len(audio) < self.dim_2[1]:
                wave = np.array(np.pad(audio, (0, self.dim_2[1] - len(audio)), 'constant', constant_values= 0))
            else:
                wave = audio[:self.dim_2[1]]
              
            X_2[i,] = wave
            
            #Store class
            y[i] = self.all_classes.index(cla)
            
            X = [X_1, X_2]
            

        return X, keras.utils.to_categorical(y, num_classes=len(self.all_classes))
                   
                        
    
    def __getitem__(self, index):
        """ Generate one batch of data """
         
        # Generates indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)     
        return X, y
                   
                   
    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1  
        return result 
          
        
                   
                   
    
    
    
            
