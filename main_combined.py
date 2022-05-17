import pandas as pd
import numpy as np
import keras
import os
from keras import backend as K
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import class_weight
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from mixup_layer import MixupLayer
import metrics
from scipy.signal import correlate
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.stats import hmean
from sklearn.linear_model import LogisticRegression


class LogMelSpectrogram(tf.keras.layers.Layer):
    """
    Compute log-magnitude mel-scaled spectrograms.
    https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
    """

    def __init__(self, sample_rate, fft_size, hop_size, n_mels,
                 f_min=0.0, f_max=None, **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=fft_size//2+1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of log-mel-spectrograms
        """
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def power_to_db(magnitude, amin=1e-10, top_db=80.0):
            """
            https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
            """
            ref_value = 1.0#tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            #log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=False)

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                     self.mel_filterbank)

        log_mel_spectrograms = power_to_db(mel_spectrograms)

        # add channel dimension
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)

        return log_mel_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'n_mels': self.n_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(LogMelSpectrogram, self).get_config())

        return config


def mixupLoss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true=y_pred[:, :, 1], y_pred=y_pred[:, :, 0])


def make_mean(mat, label):
    label, index = np.unique(label, return_inverse=True)
    mean = []
    mat = np.array(mat)
    for i, spk in enumerate(label):
        mean.append(np.mean(mat[np.nonzero(index == i)], axis=0))
    mean = length_norm(mean)
    return mean, label


def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line/np.math.sqrt(sum(np.power(line, 2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


def model_xvector_cnn(num_classes, raw_dim, n_subclusters):
    data_input = tf.keras.layers.Input(shape=(raw_dim, 1), dtype='float32')
    label_input = tf.keras.layers.Input(shape=(num_classes), dtype='float32')
    y = label_input
    x = data_input
    l2_weight_decay = tf.keras.regularizers.l2(1e-5)
    x_mix, y = MixupLayer(prob=1)([x, y])
    #x = tf.keras.layers.GaussianNoise(0.2)(x)
    # FFT
    #x = tf.keras.layers.Reshape((int(raw_dim/2),2))(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.abs(tf.signal.fft(tf.complex(x[:,:,0], tf.zeros_like(x[:,:,0])))[:,:int(raw_dim/2)]))(x_mix)
    x = tf.keras.layers.Reshape((-1,1))(x)
    #x = tf.keras.layers.BatchNormalization(-2)(x)
    x = tf.keras.layers.Conv1D(128, 256, strides=64, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.MaxPooling1D(256, strides=64)(x)
    x = tf.keras.layers.Conv1D(128, 64, strides=32, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 16, strides=4, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    #x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)

    #x = tf.keras.layers.Reshape((raw_dim,))(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)

    emb_fft = tf.keras.layers.Dense(128, name='emb_fft', kernel_regularizer=l2_weight_decay, use_bias=False)(x)

    # LOG-MEL
    x = tf.keras.layers.Reshape((160000,))(x_mix)
    x = LogMelSpectrogram(16000, 1024, 256, 128, f_max=8000)(x)
    #x = tf.keras.layers.Lambda(lambda x: x-tf.math.reduce_mean(tf.math.reduce_mean(x, axis=1, keepdims=True), axis=0, keepdims=True))(x)
    x = tf.keras.layers.BatchNormalization(axis=-2)(x)
    x = tf.keras.layers.Lambda(lambda x: x - tf.math.reduce_mean(x, axis=1, keepdims=True))(x)
    # first block
    x = tf.keras.layers.Conv2D(16, 7, strides=2, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    #x = tf.keras.layers.BatchNormalization(-2)(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
    # x, y = MixupLayer(prob=1)([x, y])

    # second block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    #xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])
    #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    # x, y = MixupLayer(prob=1)([x, y])

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    #xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])
    #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    # x, y = MixupLayer(prob=1)([x, y])
    """
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    #xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    #x, y = MixupLayer(prob=1)([x, y])
    """
    # third block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    #xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=32, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.Add()([x, xr])
    #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    # x, y = MixupLayer(prob=1)([x, y])

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    #xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])
    #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    # x, y = MixupLayer(prob=1)([x, y])
    """
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    #xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    #x, y = MixupLayer(prob=1)([x, y])

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    #xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    #x, y = MixupLayer(prob=1)([x, y])
    """
    # fourth block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    #xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=64, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.Add()([x, xr])
    #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    # x, y = MixupLayer(prob=1)([x, y])

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    #xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])
    #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    # x, y = MixupLayer(prob=1)([x, y])
    """
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    #xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    #x, y = MixupLayer(prob=1)([x, y])

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    #xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    #x, y = MixupLayer(prob=1)([x, y])

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    #xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    #x, y = MixupLayer(prob=1)([x, y])

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    #xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    #x, y = MixupLayer(prob=1)([x, y])
    """
    # fifth block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    #xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=128, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.Add()([x, xr])
    #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    # x, y = MixupLayer(prob=1)([x, y])

    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    #xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])
    #x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Dropout(rate=0.2)(x)
    # x, y = MixupLayer(prob=1)([x, y])
    """
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    # xr, y = MixupLayer(prob=1)([xr, y])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.BatchNormalization()(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay)(xr)
    x = tf.keras.layers.Add()([x, xr])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    #x, y = MixupLayer(prob=1)([x, y])
    """
    """
    # GRU layer + attention
    x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=2))(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, -1))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(x)
    query, value = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=2))(x)
    x = tf.keras.layers.Attention(name='Attention')([query, value])
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    x, y = MixupLayer(prob=1)([x, y])
    """
    # x = tf.keras.layers.MaxPooling2D((8, 1), padding='same')(x)
    # mean = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1))(x)
    # std = tf.keras.layers.Lambda(lambda x: K.std(x, axis=1))(x)  # this causes numerical issues!!!!
    # stack = tf.keras.layers.concatenate([mean, std])

    x = tf.keras.layers.MaxPooling2D((10, 1), padding='same')(x)
    x = tf.keras.layers.Flatten(name='flat')(x)

    # x = tf.keras.layers.Reshape((80, 128))(x)
    # x = tf.keras.layers.Permute((2,1))(x)
    emb_mel = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, name='emb_mel', use_bias=False)(x)

    # combine embeddings
    #s = tf.keras.layers.Dense(2, activation='softmax', name='emb_wts')(y)
    #x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[2][:,0],axis=-1)*x[0]+tf.expand_dims(x[2][:,1], axis=-1)*x[1], name='emb')([emb_fft, emb_mel, s])
    #x = tf.keras.layers.Lambda(lambda x: x[1], name='emb')([emb_fft, emb_mel, s])
    x = tf.keras.layers.Concatenate(axis=-1)([emb_fft, emb_mel])
    output = metrics.AdaCos(n_classes=num_classes, n_subclusters=n_subclusters)([x, y, label_input])

    #output_mel = metrics.AdaCos(n_classes=num_classes, n_subclusters=n_subclusters)([emb_mel, y, label_input])
    #output_fft = metrics.AdaCos(n_classes=num_classes, n_subclusters=n_subclusters)([emb_fft, y, label_input])
    #output = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[2][:,0],axis=-1)*x[0]+tf.expand_dims(x[2][:,1], axis=-1)*x[1], name='emb')([output_fft, output_mel, s])
    #output = tf.keras.layers.Average()([output_mel,output_fft])
    loss_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([output, y])

    return data_input, label_input, loss_output


########################################################################################################################
# Load data and compute embeddings
########################################################################################################################
target_sr = 16000

# load train data
print('Loading train data')
categories = os.listdir("./dev_data")

if os.path.isfile(str(target_sr) + '_train_raw.npy'):
    train_raw = np.load(str(target_sr) + '_train_raw.npy')
    train_ids = np.load('train_ids.npy')
    train_files = np.load('train_files.npy')
    train_atts = np.load('train_atts.npy')
    train_domains = np.load('train_domains.npy')
else:
    train_raw = []
    train_ids = []
    train_files = []
    train_atts = []
    train_domains = []
    dicts = ['./dev_data/', './eval_data/']
    #dicts = ['./eval_data/']
    #dicts = ['./dev_data/']
    eps=1e-12
    for label, category in enumerate(categories):
        print(category)
        for dict in dicts:
            for count, file in tqdm(enumerate(os.listdir(dict + category + "/train")), total=len(os.listdir(dict + category + "/train"))):
                file_path = dict + category + "/train/" + file
                wav, fs = sf.read(file_path)
                raw = librosa.core.to_mono(wav.transpose()).transpose()[:10*target_sr]
                train_raw.append(raw)
                train_ids.append(category + '_' + file.split('_')[1])
                train_files.append(file_path)
                train_domains.append(file.split('_')[2])
                train_atts.append('_'.join(file.split('.wav')[0].split('_')[6:]))
    # reshape arrays and store
    train_ids = np.array(train_ids)
    train_files = np.array(train_files)
    train_raw = np.expand_dims(np.array(train_raw, dtype=np.float32), axis=-1)
    train_atts = np.array(train_atts)
    train_domains = np.array(train_domains)
    np.save('train_ids.npy', train_ids)
    np.save('train_files.npy', train_files)
    np.save('train_atts.npy', train_atts)
    np.save('train_domains.npy', train_domains)
    np.save(str(target_sr) + '_train_raw.npy', train_raw)

# load evaluation data
print('Loading evaluation data')
if os.path.isfile(str(target_sr) +  '_eval_raw.npy'):
    eval_raw = np.load(str(target_sr) +  '_eval_raw.npy')
    eval_ids = np.load('eval_ids.npy')
    eval_normal = np.load('eval_normal.npy')
    eval_files = np.load('eval_files.npy')
    eval_atts = np.load('eval_atts.npy')
    eval_domains = np.load('eval_domains.npy')
else:
    eval_raw = []
    eval_ids = []
    eval_normal = []
    eval_files = []
    eval_atts = []
    eval_domains = []
    eps=1e-12
    for label, category in enumerate(categories):
        print(category)
        for count, file in tqdm(enumerate(os.listdir("./dev_data/" + category + "/test")), total=len(os.listdir("./dev_data/" + category + "/test"))):
            file_path = "./dev_data/" + category + "/test/" + file
            wav, fs = sf.read(file_path)
            raw = librosa.core.to_mono(wav.transpose()).transpose()[:10*target_sr]
            eval_raw.append(raw)
            eval_ids.append(category + '_' + file.split('_')[1])
            eval_normal.append(file.split('_test_')[1].split('_')[0] == 'normal')
            eval_files.append(file_path)
            eval_domains.append(file.split('_')[2])
            eval_atts.append('_'.join(file.split('.wav')[0].split('_')[6:]))
    # reshape arrays and store
    eval_ids = np.array(eval_ids)
    eval_normal = np.array(eval_normal)
    eval_files = np.array(eval_files)
    eval_atts = np.array(eval_atts)
    eval_domains = np.array(eval_domains)
    eval_raw = np.expand_dims(np.array(eval_raw, dtype=np.float32), axis=-1)
    np.save('eval_ids.npy', eval_ids)
    np.save('eval_normal.npy', eval_normal)
    np.save('eval_files.npy', eval_files)
    np.save('eval_atts.npy', eval_atts)
    np.save('eval_domains.npy', eval_domains)
    np.save(str(target_sr) + '_eval_raw.npy', eval_raw)

"""
# load test data
print('Loading test data')
if os.path.isfile(str(target_sr) + '_test_raw.npy'):
    test_raw = np.load(str(target_sr) + '_test_raw.npy')
    test_ids = np.load('test_ids.npy')
    test_files = np.load('test_files.npy')
else:
    test_raw = []
    test_ids = []
    test_files = []
    eps = 1e-12
    for label, category in enumerate(categories):
        print(category)
        for count, file in tqdm(enumerate(os.listdir("./eval_data/" + category + "/test")), total=len(os.listdir("./eval_data/" + category + "/test"))):
            file_path = "./eval_data/" + category + "/test/" + file
            wav, fs = sf.read(file_path)
            raw = librosa.core.to_mono(wav.transpose()).transpose()[:10*target_sr]
            test_raw.append(raw)
            test_ids.append(category + '_' + file.split('_')[-2])
            test_files.append(file_path)
    # reshape arrays and store
    test_ids = np.array(test_ids)
    test_files = np.array(test_files)
    test_raw = np.expand_dims(np.array(test_raw, dtype=np.float32), axis=-1)
    np.save('test_ids.npy', test_ids)
    np.save('test_files.npy', test_files)
    np.save(str(target_sr) + '_test_raw.npy', test_raw)
"""
# encode ids as labels
le_4train = LabelEncoder()
#le.fit(np.concatenate([np.unique(train_ids), np.unique(eval_ids), np.unique(test_ids)]))
train_ids_4train = np.array(['_'.join([train_ids[k], train_atts[k]]) for k in np.arange(train_ids.shape[0])])
eval_ids_4train = np.array(['_'.join([eval_ids[k], eval_atts[k]]) for k in np.arange(eval_ids.shape[0])])

le_4train.fit(np.concatenate([train_ids_4train,eval_ids_4train], axis=0))
num_classes = len(np.unique(np.concatenate([train_ids_4train,eval_ids_4train], axis=0)))
train_labels_4train = le_4train.transform(train_ids_4train)
eval_labels_4train = le_4train.transform(eval_ids_4train)
#test_labels = le.transform(test_ids)

le = LabelEncoder()
train_labels = le.fit_transform(train_ids)
eval_labels = le.transform(eval_ids)
num_classes = len(np.unique(train_labels))

# distinguish between normal and anomalous samples
unknown_raw = eval_raw[~eval_normal]
unknown_labels = eval_labels[~eval_normal]
unknown_labels_4train = eval_labels_4train[~eval_normal]
unknown_files = eval_files[~eval_normal]
unknown_ids = eval_ids[~eval_normal]
unknown_domains = eval_domains[~eval_normal]
eval_raw = eval_raw[eval_normal]
eval_labels = eval_labels[eval_normal]
eval_labels_4train = eval_labels_4train[eval_normal]
eval_files = eval_files[eval_normal]
eval_ids = eval_ids[eval_normal]
eval_domains = eval_domains[eval_normal]

#print(le.inverse_transform(np.unique(train_labels)))
#print(le_4train.inverse_transform(np.unique(train_labels_4train)))
"""
# feature normalization
print('Normalizing data')
eps = 1e-12
# predicting with GMMs
pred_eval = np.zeros((eval_raw.shape[0], np.unique(train_labels).shape[0]))
pred_unknown = np.zeros((unknown_raw.shape[0], np.unique(train_labels).shape[0]))
#pred_test = np.zeros((test_raw.shape[0], np.unique(train_labels).shape[0]))
pred_train = np.zeros((train_raw.shape[0], np.unique(train_labels).shape[0]))
for lab in np.unique(train_labels):
    mean_raw = np.mean(train_raw[train_labels==lab], axis=0, keepdims=True)
    std_raw = np.std(train_raw[train_labels==lab], axis=0, keepdims=True)
    train_raw[train_labels==lab] = (train_raw[train_labels==lab]-mean_raw)/(std_raw+eps)
    if np.sum(eval_labels==lab)>0:
        eval_raw[eval_labels==lab] = (eval_raw[eval_labels==lab]-mean_raw)/(std_raw+eps)
        unknown_raw[unknown_labels==lab] = (unknown_raw[unknown_labels==lab]-mean_raw)/(std_raw+eps)
    #if np.sum(test_labels==lab)>0:
    #    test_raw[test_labels==lab] = (test_raw[test_labels==lab]-mean_raw)/(std_raw+eps)
"""
"""
for lab in np.unique(train_labels):
    print(le.inverse_transform([lab])[0])
    plt.subplot(3,3,1)
    plt.plot(np.mean(train_raw[train_labels==lab,:,0],axis=0))
    plt.subplot(3,3,2)
    plt.plot(np.mean(train_raw[train_labels==lab,:,0],axis=0)-np.mean(train_raw[train_labels==lab,:,0],axis=0))
    plt.subplot(3,3,3)
    plt.plot(np.sum(np.abs(np.mean(train_raw[train_labels==lab,:,0],axis=0, keepdims=True)-train_raw[train_labels==lab,:,0]),axis=1))
    plt.subplot(3,3,4)
    plt.plot(np.mean(eval_raw[eval_labels==lab,:,0],axis=0))
    plt.subplot(3,3,5)
    plt.plot(np.mean(train_raw[train_labels==lab,:,0],axis=0)-np.mean(eval_raw[eval_labels==lab,:,0],axis=0))
    plt.subplot(3,3,6)
    plt.plot(np.sum(np.abs(np.mean(train_raw[train_labels==lab,:,0],axis=0, keepdims=True)-eval_raw[eval_labels==lab,:,0]),axis=1))
    plt.subplot(3,3,7)
    plt.plot(np.mean(unknown_raw[unknown_labels==lab,:,0],axis=0))
    plt.subplot(3,3,8)
    plt.plot(np.mean(train_raw[train_labels==lab,:,0],axis=0)-np.mean(unknown_raw[unknown_labels==lab,:,0],axis=0))
    plt.subplot(3,3,9)
    plt.plot(np.sum(np.abs(np.mean(train_raw[train_labels==lab,:,0],axis=0, keepdims=True)-unknown_raw[unknown_labels==lab,:,0]),axis=1))
    plt.show()
    plt.plot(np.sum(np.abs(np.mean(train_raw[train_labels==lab,:,0],axis=0, keepdims=True)-train_raw[train_labels==lab,:,0]),axis=1),'.')
    plt.plot(np.sum(np.abs(np.mean(train_raw[train_labels==lab,:,0],axis=0, keepdims=True)-eval_raw[eval_labels==lab,:,0]),axis=1),'.')
    plt.plot(np.sum(np.abs(np.mean(train_raw[train_labels==lab,:,0],axis=0, keepdims=True)-unknown_raw[unknown_labels==lab,:,0]),axis=1),'.')
    plt.show()
"""

########################################################################################################################
# train x-vector cnn on train partition of development set
########################################################################################################################
batch_size = 64
batch_size_test = 64
epochs = 100
aeons = 4
alpha = 1

# predicting with GMMs
pred_eval = np.zeros((eval_raw.shape[0], np.unique(train_labels_4train).shape[0]))
pred_unknown = np.zeros((unknown_raw.shape[0], np.unique(train_labels_4train).shape[0]))
#pred_test = np.zeros((test_raw.shape[0], np.unique(train_labels_4train).shape[0]))
pred_train = np.zeros((train_labels.shape[0], np.unique(train_labels_4train).shape[0]))
source_train = np.array([file.split('_')[3]=='source' for file in train_files.tolist()])
source_eval = np.array([file.split('_')[3]=='source' for file in eval_files.tolist()])
source_unknown = np.array([file.split('_')[3]=='source' for file in unknown_files.tolist()])

for n_subclusters in [16]:#2**np.arange(6):
    #n_subclusters=32
    y_train_cat = keras.utils.np_utils.to_categorical(train_labels, num_classes=num_classes)
    y_eval_cat = keras.utils.np_utils.to_categorical(eval_labels, num_classes=num_classes)
    y_unknown_cat = keras.utils.np_utils.to_categorical(unknown_labels, num_classes=num_classes)
    #y_test_cat = keras.utils.np_utils.to_categorical(test_labels, num_classes=num_classes)

    # compile model
    data_input, label_input, loss_output = model_xvector_cnn(num_classes=num_classes,
                                                             raw_dim=eval_raw.shape[1], n_subclusters=n_subclusters)
    model = tf.keras.Model(inputs=[data_input, label_input], outputs=[loss_output])
    model.compile(loss=[mixupLoss], optimizer=tf.keras.optimizers.Adam())
    print(model.summary())
    """
    x_vector_model = tf.keras.Model(model.input, model.layers[-3].output)
    wts_init = np.zeros((num_classes*n_subclusters, 256))
    from sklearn.cluster import KMeans
    for k_lab, lab in tqdm(enumerate(np.unique(train_labels)), total=len(np.unique(train_labels))):
        train_embs = x_vector_model.predict([train_raw[train_labels==lab], y_train_cat[train_labels==lab]], batch_size=batch_size)
        #wts_init = x_vector_model.predict([np.random.rand(num_classes*n_subclusters, 628, 128, 1), np.zeros((41, 1))], batch_size=batch_size)
        kmeans = KMeans(n_clusters=n_subclusters, random_state=0).fit(train_embs)
        wts_init[k_lab*n_subclusters:(k_lab+1)*n_subclusters] = kmeans.cluster_centers_
    #model.layers[-2].set_weights([wts_init, model.layers[-2].get_weights()[1]])
    model.layers[-2].set_weights([wts_init.transpose(), model.layers[-2].get_weights()[1]])
    """
    # create data generator for mixup and random erasing for every batch
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #height_shift_range=0.99,
        #fill_mode='wrap'
    )
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join("logs"), histogram_freq=0, write_graph=True,
                                       write_images=False)
    ]

    for k in np.arange(aeons):
        print('subclusters: ' + str(n_subclusters))
        print('aeon: ' + str(k))
        # fit model
        weight_path = 'wts_raw_' + str(k + 1) + 'k_' + str(target_sr) + '_' + str(n_subclusters) + '_with_eval_no-bias_fixed-means_no-batch-norm.h5' # '_emb_distr_raw.h5'
        if not os.path.isfile(weight_path):
            class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
            #class_weights = np.sum(y_train_cat,axis=0)/np.sum(y_train_cat)
            class_weights = {i: class_weights[i] for i in range(class_weights.shape[0])}
            for k_epoch in [1]:#np.arange(epochs):
                #train_raw = np.load(str(target_sr) + '_train_raw.npy')
                #train_raw_mixed = train_raw  # np.zeros(train_raw.shape)
                #for lab in tqdm(np.unique(train_labels)):
                #    rand_target_idcs = np.random.randint(0,np.sum(~source_train*(train_labels==lab)),np.sum(source_train*(train_labels==lab)))
                #    rand_mix_coeffs = 0.5#np.random.rand(len(rand_target_idcs),1,1)
                #    train_raw[source_train*(train_labels==lab)] *= rand_mix_coeffs
                #    train_raw[source_train * (train_labels == lab)] +=(1-rand_mix_coeffs)*train_raw[~source_train*(train_labels==lab)][rand_target_idcs]
                model.fit(#[train_raw[source_train], y_train_cat[source_train]], y_train_cat[source_train], verbose=1,
                                    [train_raw, y_train_cat], y_train_cat, verbose=1,
                                    batch_size= batch_size, epochs=epochs, callbacks=callbacks,
                                    validation_data=([eval_raw, y_eval_cat], y_eval_cat))#, class_weight=class_weights)
                #train_raw = None
            model.save(weight_path)
        else:
            model = tf.keras.models.load_model(weight_path,
                                               custom_objects={'MixupLayer': MixupLayer, 'mixupLoss': mixupLoss, 'AdaCos': metrics.AdaCos, 'LogMelSpectrogram': LogMelSpectrogram})

        #x_vector_model = tf.keras.Model(model.input, model.get_layer('emb').output)
        x_vector_model = tf.keras.Model(model.input, model.layers[-3].output) # -3
        eval_x_vecs = x_vector_model.predict([eval_raw, y_eval_cat], batch_size=batch_size)
        train_x_vecs = x_vector_model.predict([train_raw, y_train_cat], batch_size=batch_size)
        unknown_x_vecs = x_vector_model.predict([unknown_raw, y_unknown_cat], batch_size=batch_size)
        #test_x_vecs = x_vector_model.predict([test_raw, y_test_cat], batch_size=batch_size)

        # length normalization
        print('normalizing lengths')
        x_train_ln = length_norm(train_x_vecs)
        x_eval_ln = length_norm(eval_x_vecs)
        #x_test_ln = length_norm(test_x_vecs)
        x_unknown_ln = length_norm(unknown_x_vecs)

        model_means = model.layers[-2].get_weights()[0].transpose()
        model_means_ln = length_norm(model_means)
        """
        dm = np.dot(model_means_ln, x_train_ln.transpose())
        print(dm.shape)
        #plt.imshow(dm, aspect='auto')
        #plt.show()
        #plt.plot(np.argmax(dm, axis=0), '.')
        #plt.show()
        plt.imshow(x_eval_ln[source_eval].transpose(), aspect='auto')
        plt.show()
        plt.imshow(np.dot(model_means_ln, np.concatenate([x_eval_ln[source_eval], x_unknown_ln[source_unknown]], axis=0).transpose()), aspect='auto')
        #plt.plot(np.max(np.dot(model_means_ln, x_unknown_ln[source_unknown].transpose()), axis=0), '.')
        plt.show()
        """
        #x_train_ln = np.mean(train_x_vecs[:,:,:,0], axis=1)
        #x_eval_ln = np.mean(eval_x_vecs[:,:,:,0], axis=1)
        #x_unknown_ln = np.mean(unknown_x_vecs[:,:,:,0], axis=1)
        """
        for j, lab in tqdm(enumerate(np.unique(train_labels)), total=len(np.unique(train_labels))):
            if np.sum(eval_labels == lab) > 0:
                print(le.inverse_transform([lab])[0])
                # mix samples
                train_mixed = np.zeros((np.sum((train_labels==lab)*~source_train)*np.sum((train_labels==lab)*source_train),train_raw.shape[1],1))
                for j_target in np.arange(np.sum((train_labels==lab)*~source_train)):
                    train_mixed[j_target*np.sum((train_labels==lab)*source_train):(j_target+1)*np.sum((train_labels==lab)*source_train)] = 0.5*train_raw[(train_labels==lab)*source_train]+0.5*np.expand_dims(train_raw[(train_labels==lab)*~source_train][j_target], axis=0)
                # extract embeddings
                train_mixed_embs = x_vector_model.predict([train_mixed, np.zeros((train_mixed.shape[0],y_train_cat.shape[1]))], batch_size=batch_size)
                train_mixed_embs_ln = length_norm(train_mixed_embs)
                # train GMM
                clf1 = GaussianMixture(n_components=n_subclusters, covariance_type='full', reg_covar=1e-3).fit(train_mixed_embs_ln)
                train_mixed = None
                # compute scores
                tmp_mixed = np.zeros(train_raw[train_labels==lab].shape)
                pred_eval_tmp = np.zeros((np.sum(eval_labels==lab),1))
                pred_unknown_tmp = np.zeros((np.sum(unknown_labels==lab),1))
                for j_sample in tqdm(np.arange(np.sum(eval_labels==lab))):
                    # mix with training samples
                    tmp_mixed = train_raw[train_labels==lab]*0.5+0.5*np.expand_dims(eval_raw[eval_labels==lab][j_sample], axis=0)
                    # get embeddings
                    eval_mixed_embs = x_vector_model.predict([tmp_mixed, np.zeros((tmp_mixed.shape[0], y_train_cat.shape[1]))], batch_size=batch_size)
                    eval_mixed_embs_ln = length_norm(eval_mixed_embs)
                    # evaluate GMM
                    scores_eval_mixed = -clf1.score_samples(eval_mixed_embs_ln)
                    # take mean of scores
                    pred_eval_tmp[j_sample] = np.mean(scores_eval_mixed)
                for j_sample in tqdm(np.arange(np.sum(unknown_labels == lab))):
                    # mix with training samples
                    tmp_mixed = train_raw[train_labels==lab] * 0.5 + 0.5 * np.expand_dims(unknown_raw[unknown_labels == lab][j_sample], axis=0)
                    # get embeddings
                    unknown_mixed_embs = x_vector_model.predict(
                        [tmp_mixed, np.zeros((tmp_mixed.shape[0], y_train_cat.shape[1]))], batch_size=batch_size)
                    unknown_mixed_embs_ln = length_norm(unknown_mixed_embs)
                    # evaluate GMM
                    scores_unknown_mixed = -clf1.score_samples(unknown_mixed_embs_ln)
                    # take mean of scores
                    pred_unknown_tmp[j_sample] = np.mean(scores_unknown_mixed)
                tmp_mixed = None
    
                # plt.plot(pred_train[train_labels_4train == lab, j],'.')
                plt.plot(pred_eval_tmp,'.')
                plt.plot(pred_unknown_tmp,'.')
                plt.show()
                auc = roc_auc_score(
                    np.concatenate([np.zeros(np.sum(eval_labels == lab)), np.ones(np.sum(unknown_labels == lab))], axis=0),
                    np.concatenate([pred_eval_tmp, pred_unknown_tmp], axis=0))
                print('AUC: ' + str(auc))
                auc = roc_auc_score(
                    np.concatenate([np.zeros(np.sum((eval_labels == lab)*source_eval)), np.ones(np.sum((unknown_labels == lab)*source_unknown))], axis=0),
                    np.concatenate([pred_eval_tmp[source_eval[eval_labels==lab]], pred_unknown_tmp[source_unknown[unknown_labels==lab]]], axis=0))
                print('AUC for source domain: ' + str(auc))
                auc = roc_auc_score(
                    np.concatenate([np.zeros(np.sum((eval_labels == lab)*~source_eval)), np.ones(np.sum((unknown_labels == lab)*~source_unknown))], axis=0),
                    np.concatenate([pred_eval_tmp[~source_eval[eval_labels==lab]], pred_unknown_tmp[~source_unknown[unknown_labels==lab]]], axis=0))
                print('AUC for target domain: ' + str(auc))
                pred_eval[eval_labels == lab, :] = pred_eval_tmp
                pred_unknown[unknown_labels == lab, :] = pred_unknown_tmp
        """

        for n_subclusters_gmm in [15]:
            n_subclusters_gmm = n_subclusters
            for j, lab in tqdm(enumerate(np.unique(train_labels)), total=len(np.unique(train_labels))):

                #plt.imshow(np.concatenate([x_train_ln[train_labels==lab],x_eval_ln[eval_labels==lab], x_unknown_ln[unknown_labels==lab]],axis=0).transpose(), aspect='auto')
                #plt.show()
                #clf1 = GaussianMixture(n_components=n_subclusters_gmm, covariance_type='full', reg_covar=1e-3, means_init=model_means_ln[j * n_subclusters:(j + 1) * n_subclusters]).fit(x_train_ln[train_labels==lab])#.fit(x_train_ln[train_labels==lab][source_train[train_labels==lab]])
                if np.sum(train_labels == lab) > 1:
                    if np.sum(train_labels == lab) >= n_subclusters_gmm:
                        clf1 = GaussianMixture(n_components=n_subclusters_gmm, covariance_type='full', reg_covar=1e-3).fit(
                            x_train_ln[train_labels == lab])
                    else:
                        clf1 = GaussianMixture(n_components=np.sum(train_labels == lab), covariance_type='full',
                                               reg_covar=1e-3).fit(x_train_ln[train_labels == lab])
                    pred_train[train_labels==lab, :] += np.expand_dims(-clf1.score_samples(x_train_ln[train_labels==lab]), axis=-1)
                    if np.sum(eval_labels==lab)>0:
                        pred_eval[eval_labels==lab, :] += np.expand_dims(-clf1.score_samples(x_eval_ln[eval_labels==lab]), axis=-1)
                        pred_unknown[unknown_labels==lab, :] += np.expand_dims(-clf1.score_samples(x_unknown_ln[unknown_labels==lab]), axis=-1)

                    #pred_train += np.expand_dims(-clf1.score_samples(x_train_ln), axis=-1)
                    #if np.sum(eval_labels == lab) > 0:
                    #    pred_eval += np.expand_dims(-clf1.score_samples(x_eval_ln), axis=-1)
                    #    pred_unknown += np.expand_dims(-clf1.score_samples(x_unknown_ln), axis=-1)
                    #pred_train[:, j] += -clf1.score_samples(x_train_ln)
                    #pred_eval[:, j] += -clf1.score_samples(x_eval_ln)
                    #pred_unknown[:, j] += -clf1.score_samples(x_unknown_ln)
                    #pred_test[:, j] += -clf1.score_samples(x_test_ln)
                    if np.sum(eval_labels==lab)>0:
                        auc = roc_auc_score(np.concatenate([np.zeros(np.sum(eval_labels==lab)), np.ones(np.sum(unknown_labels==lab))], axis=0),
                                            np.concatenate([pred_eval[eval_labels == lab, j], pred_unknown[unknown_labels == lab, j]], axis=0))

                        #plt.plot(pred_train[train_labels_4train == lab, j],'.')
                        #plt.plot(pred_eval[eval_labels_4train == lab, j],'.')
                        #plt.plot(pred_unknown[unknown_labels_4train == lab, j],'.')
                        #plt.show()
                        print('AUC with mean: ' + str(auc))

        for n_subclusters_gmm in [15]:
            n_subclusters_gmm = n_subclusters
            for j, lab in tqdm(enumerate(np.unique(train_labels_4train)), total=len(np.unique(train_labels_4train))):

                #plt.imshow(np.concatenate([x_train_ln[train_labels==lab],x_eval_ln[eval_labels==lab], x_unknown_ln[unknown_labels==lab]],axis=0).transpose(), aspect='auto')
                #plt.show()
                #clf1 = GaussianMixture(n_components=n_subclusters_gmm, covariance_type='full', reg_covar=1e-3, means_init=model_means_ln[j * n_subclusters:(j + 1) * n_subclusters]).fit(x_train_ln[train_labels==lab])#.fit(x_train_ln[train_labels==lab][source_train[train_labels==lab]])
                if np.sum(train_labels_4train == lab) > 1:
                    if np.sum(train_labels_4train == lab) >= n_subclusters_gmm:
                        clf1 = GaussianMixture(n_components=n_subclusters_gmm, covariance_type='full', reg_covar=1e-3).fit(
                            x_train_ln[train_labels_4train == lab])
                    else:
                        clf1 = GaussianMixture(n_components=np.sum(train_labels_4train == lab), covariance_type='full',
                                               reg_covar=1e-3).fit(x_train_ln[train_labels_4train == lab])
                    lab_sec = le.transform([le_4train.inverse_transform([lab])[0].split('_')[0]+'_'+le_4train.inverse_transform([lab])[0].split('_')[1]])[0]
                    pred_train[train_labels==lab_sec, j] += -clf1.score_samples(x_train_ln[train_labels==lab_sec])
                    if np.sum(eval_labels == lab_sec) > 0:
                        pred_eval[eval_labels==lab_sec, j] += -clf1.score_samples(x_eval_ln[eval_labels==lab_sec])
                        pred_unknown[unknown_labels==lab_sec, j] += -clf1.score_samples(x_unknown_ln[unknown_labels==lab_sec])
                    #pred_train[:, j] += -clf1.score_samples(x_train_ln)
                    #pred_eval[:, j] += -clf1.score_samples(x_eval_ln)
                    #pred_unknown[:, j] += -clf1.score_samples(x_unknown_ln)
                    #pred_test[:, j] += -clf1.score_samples(x_test_ln)
                    if np.sum(eval_labels_4train==lab)>0:
                        auc = roc_auc_score(np.concatenate([np.zeros(np.sum(eval_labels_4train==lab)), np.ones(np.sum(unknown_labels_4train==lab))], axis=0),
                                            np.concatenate([pred_eval[eval_labels_4train == lab, j], pred_unknown[unknown_labels_4train == lab, j]], axis=0))

                        #plt.plot(pred_train[train_labels_4train == lab, j],'.')
                        #plt.plot(pred_eval[eval_labels_4train == lab, j],'.')
                        #plt.plot(pred_unknown[unknown_labels_4train == lab, j],'.')
                        #plt.show()
                        print('AUC with mean: ' + str(auc))
            j=0
            pred_eval_plda = pred_eval
            pred_unknown_plda = pred_unknown
            #plt.plot(pred_eval_plda,'.')
            #plt.plot(pred_unknown_plda, '.')
            #plt.show()

            #pred_test_plda = pred_test

            # apply logistic regression to combine and calibrate scores
            #lr = LogisticRegression(class_weight='balanced', max_iter=10000, solver='saga', n_jobs=-1).fit(pred_train, train_labels)
            #pred_eval_plda = -lr.predict_proba(pred_eval_plda)
            #pred_unknown_plda = -lr.predict_proba(pred_unknown_plda)
            """
            # output performance
            print('performance on evaluation set')
            y_pred_eval = np.argmin(pred_eval_plda, axis=1)
            y_pred_unknown = np.argmin(pred_unknown_plda, axis=1)
            print('####################')
            print('closed-set performance by machine id:')
            print('evaluation files: ' + str(np.mean(y_pred_eval == eval_labels)))
            print('unknown files: ' + str(np.mean(y_pred_unknown == unknown_labels)))
            print('all files: ' + str(
                np.mean(np.hstack([y_pred_unknown, y_pred_eval]) == np.hstack([unknown_labels, eval_labels]))))
            print('####################')
            type_labels_eval1 = np.array([eval_id.split('_')[0] for eval_id in eval_ids])
            type_labels_unknown1 = np.array([unknown_id.split('_')[0] for unknown_id in unknown_ids])
            type_pred_eval1 = np.array([pred_id.split('_')[0] for pred_id in le.inverse_transform(y_pred_eval)])
            type_pred_unknown1 = np.array([pred_id.split('_')[0] for pred_id in le.inverse_transform(y_pred_unknown)])
            print('closed-set performance by machine type:')
            print('evaluation files: ' + str(np.mean(type_pred_eval1 == type_labels_eval1)))
            print('unknown files: ' + str(np.mean(type_pred_unknown1 == type_labels_unknown1)))
            print('all files: ' + str(np.mean(
                np.hstack([type_pred_unknown1, type_pred_eval1]) == np.hstack([type_labels_unknown1, type_labels_eval1]))))
            print('####################')
    
            print('closed-set performance on test data')
            y_pred_test = np.argmin(pred_test_plda, axis=1)
            type_labels_test1 = np.array([test_id.split('_')[0] for test_id in test_ids])
            type_pred_test1 = np.array([pred_id.split('_')[0] for pred_id in le.inverse_transform(y_pred_test)])
            print('for machine id: ' + str(np.mean(y_pred_test == test_labels)))
            print('for machine type: ' + str(np.mean(type_pred_test1 == type_labels_test1)))
            print('####################')
            """
            aucs = []
            p_aucs = []
            for j, cat in enumerate(np.unique(eval_ids)):
                #y_pred = np.concatenate([pred_eval_plda[eval_labels == le.transform([cat]), le.transform([cat])],
                #                         pred_unknown_plda[unknown_labels == le.transform([cat]), le.transform([cat])]],
                #                        axis=0)
                y_pred = np.mean(np.concatenate([pred_eval_plda[eval_labels == le.transform([cat])],
                                         pred_unknown_plda[unknown_labels == le.transform([cat])]],
                                        axis=0), axis=-1)
                y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))),
                                         np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
                auc = roc_auc_score(y_true, y_pred)
                aucs.append(auc)
                p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
                p_aucs.append(p_auc)
                print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
                print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))

                source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])], source_unknown[unknown_labels == le.transform([cat])]], axis=0)
                auc = roc_auc_score(y_true[source_all], y_pred[source_all])
                p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
                print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
                print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
                auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
                p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
                print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
                print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
            print('####################')
            aucs = np.array(aucs)
            p_aucs = np.array(p_aucs)
            for cat in categories:
                mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
                print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
                mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
                print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
            print('####################')
            for cat in categories:
                mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
                mean_p_auc =hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
                print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
            print('####################')
            mean_auc = hmean(aucs)
            print('mean AUC: ' + str(mean_auc * 100))
            mean_p_auc = hmean(p_aucs)
            print('mean pAUC: ' + str(mean_p_auc * 100))
np.save('pred_eval_sec.npy', pred_eval_plda)
np.save('pred_unknown_sec.npy', pred_unknown_plda)


pred_eval_att = np.load('pred_eval_att.npy')
pred_unknown_att = np.load('pred_unknown_att.npy')

pred_eval_plda += pred_eval_att
pred_unknown_plda += pred_unknown_att
aucs = []
p_aucs = []
for j, cat in enumerate(np.unique(eval_ids)):
    #y_pred = np.concatenate([pred_eval_plda[eval_labels == le.transform([cat]), le.transform([cat])],
    #                         pred_unknown_plda[unknown_labels == le.transform([cat]), le.transform([cat])]],
    #                        axis=0)
    y_pred = np.mean(np.concatenate([pred_eval_plda[eval_labels == le.transform([cat])],
                             pred_unknown_plda[unknown_labels == le.transform([cat])]],
                            axis=0), axis=-1)
    y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))),
                             np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
    auc = roc_auc_score(y_true, y_pred)
    aucs.append(auc)
    p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
    p_aucs.append(p_auc)
    print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))

    source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])], source_unknown[unknown_labels == le.transform([cat])]], axis=0)
    auc = roc_auc_score(y_true[source_all], y_pred[source_all])
    p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
    print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
    auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
    p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
    print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
print('####################')
aucs = np.array(aucs)
p_aucs = np.array(p_aucs)
for cat in categories:
    mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
    mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
print('####################')
for cat in categories:
    mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    mean_p_auc =hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
print('####################')
mean_auc = hmean(aucs)
print('mean AUC: ' + str(mean_auc * 100))
mean_p_auc = hmean(p_aucs)
print('mean pAUC: ' + str(mean_p_auc * 100))

pred_eval_plda = np.minimum(pred_eval,pred_eval_att)
pred_unknown_plda = np.minimum(pred_unknown,pred_unknown_att)
aucs = []
p_aucs = []
for j, cat in enumerate(np.unique(eval_ids)):
    #y_pred = np.concatenate([pred_eval_plda[eval_labels == le.transform([cat]), le.transform([cat])],
    #                         pred_unknown_plda[unknown_labels == le.transform([cat]), le.transform([cat])]],
    #                        axis=0)
    y_pred = np.mean(np.concatenate([pred_eval_plda[eval_labels == le.transform([cat])],
                             pred_unknown_plda[unknown_labels == le.transform([cat])]],
                            axis=0), axis=-1)
    y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))),
                             np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
    auc = roc_auc_score(y_true, y_pred)
    aucs.append(auc)
    p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
    p_aucs.append(p_auc)
    print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))

    source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])], source_unknown[unknown_labels == le.transform([cat])]], axis=0)
    auc = roc_auc_score(y_true[source_all], y_pred[source_all])
    p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
    print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
    auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
    p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
    print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
print('####################')
aucs = np.array(aucs)
p_aucs = np.array(p_aucs)
for cat in categories:
    mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
    mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
print('####################')
for cat in categories:
    mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    mean_p_auc =hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
print('####################')
mean_auc = hmean(aucs)
print('mean AUC: ' + str(mean_auc * 100))
mean_p_auc = hmean(p_aucs)
print('mean pAUC: ' + str(mean_p_auc * 100))

pred_eval_plda = np.maximum(pred_eval,pred_eval_att)
pred_unknown_plda = np.maximum(pred_unknown,pred_unknown_att)
aucs = []
p_aucs = []
for j, cat in enumerate(np.unique(eval_ids)):
    #y_pred = np.concatenate([pred_eval_plda[eval_labels == le.transform([cat]), le.transform([cat])],
    #                         pred_unknown_plda[unknown_labels == le.transform([cat]), le.transform([cat])]],
    #                        axis=0)
    y_pred = np.mean(np.concatenate([pred_eval_plda[eval_labels == le.transform([cat])],
                             pred_unknown_plda[unknown_labels == le.transform([cat])]],
                            axis=0), axis=-1)
    y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))),
                             np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
    auc = roc_auc_score(y_true, y_pred)
    aucs.append(auc)
    p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
    p_aucs.append(p_auc)
    print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))

    source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])], source_unknown[unknown_labels == le.transform([cat])]], axis=0)
    auc = roc_auc_score(y_true[source_all], y_pred[source_all])
    p_auc = roc_auc_score(y_true[source_all], y_pred[source_all], max_fpr=0.1)
    print('AUC for source domain of category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for source domain of category ' + str(cat) + ': ' + str(p_auc * 100))
    auc = roc_auc_score(y_true[~source_all], y_pred[~source_all])
    p_auc = roc_auc_score(y_true[~source_all], y_pred[~source_all], max_fpr=0.1)
    print('AUC for target domain of category ' + str(cat) + ': ' + str(auc * 100))
    print('pAUC for target domain of category ' + str(cat) + ': ' + str(p_auc * 100))
print('####################')
aucs = np.array(aucs)
p_aucs = np.array(p_aucs)
for cat in categories:
    mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean AUC for category ' + str(cat) + ': ' + str(mean_auc * 100))
    mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean pAUC for category ' + str(cat) + ': ' + str(mean_p_auc * 100))
print('####################')
for cat in categories:
    mean_auc = hmean(aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    mean_p_auc =hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
    print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
print('####################')
mean_auc = hmean(aucs)
print('mean AUC: ' + str(mean_auc * 100))
mean_p_auc = hmean(p_aucs)
print('mean pAUC: ' + str(mean_p_auc * 100))

# create challenge submission files
print('creating submission files')
for j, cat in enumerate(np.unique(test_ids)):
    file_idx = test_labels == le.transform([cat])
    results = pd.DataFrame()
    # results['output'] = [f.split('/')[-1] + ',' + str(s) for f, s in zip(test_files[file_idx], pred_test_plda[file_idx, le.transform([cat])])]
    results['output1'], results['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                              [str(s) for s in pred_test_plda[file_idx, le.transform([cat])]]]
    results.to_csv('teams/mfcc_emb/anomaly_score_' + cat.split('_')[0] + '_id_' + cat.split('_')[-1] + '.csv',
                   encoding='utf-8', index=False, header=False)
print('####################')
print('>>>> finished! <<<<<')
print('####################')