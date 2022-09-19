import pandas as pd
import numpy as np
import keras
import os
import soundfile as sf
import tensorflow as tf
import librosa
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from mixup_layer import MixupLayer
from subcluster_adacos import SCAdaCos
from sklearn.mixture import GaussianMixture
from scipy.stats import hmean


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
            num_spectrogram_bins=fft_size // 2 + 1,
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
            ref_value = 1.0  # tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            # log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

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


def length_norm(mat):
    norm_mat = []
    for line in mat:
        temp = line / np.math.sqrt(sum(np.power(line, 2)))
        norm_mat.append(temp)
    norm_mat = np.array(norm_mat)
    return norm_mat


def model_emb_cnn(num_classes, raw_dim, n_subclusters):
    data_input = tf.keras.layers.Input(shape=(raw_dim, 1), dtype='float32')
    label_input = tf.keras.layers.Input(shape=(num_classes), dtype='float32')
    y = label_input
    x = data_input
    l2_weight_decay = tf.keras.regularizers.l2(1e-5)
    x_mix, y = MixupLayer(prob=1)([x, y])

    # FFT
    x = tf.keras.layers.Lambda(lambda x: tf.math.abs(tf.signal.fft(tf.complex(x[:,:,0], tf.zeros_like(x[:,:,0])))[:,:int(raw_dim/2)]))(x_mix)
    x = tf.keras.layers.Reshape((-1,1))(x)
    x = tf.keras.layers.Conv1D(128, 256, strides=64, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv1D(128, 64, strides=32, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Conv1D(128, 16, strides=4, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    emb_fft = tf.keras.layers.Dense(128, name='emb_fft', kernel_regularizer=l2_weight_decay, use_bias=False)(x)

    # LOG-MEL
    x = tf.keras.layers.Reshape((160000,))(x_mix)
    x = LogMelSpectrogram(16000, 1024, 256, 128, f_max=8000)(x)
    x = tf.keras.layers.Lambda(lambda x: x - tf.math.reduce_mean(x, axis=1, keepdims=True))(x) # CMN-like normalization
    x = tf.keras.layers.BatchNormalization(axis=-2)(x)

    # first block
    x = tf.keras.layers.Conv2D(16, 7, strides=2, activation='linear', padding='same',
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)

    # second block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # third block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(32, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=32, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.Add()([x, xr])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # fourth block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=64, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.Add()([x, xr])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])

    # fifth block
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation='linear', padding='same',
                                kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(kernel_size=1, filters=128, strides=1, padding="same",
                               kernel_regularizer=l2_weight_decay, use_bias=False)(x)
    x = tf.keras.layers.Add()([x, xr])
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    xr = tf.keras.layers.LeakyReLU(alpha=0.1)(xr)
    xr = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_regularizer=l2_weight_decay, use_bias=False)(xr)
    x = tf.keras.layers.Add()([x, xr])

    x = tf.keras.layers.MaxPooling2D((20, 1), padding='same')(x)
    x = tf.keras.layers.Flatten(name='flat')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    emb_mel = tf.keras.layers.Dense(128, kernel_regularizer=l2_weight_decay, name='emb_mel', use_bias=False)(x)

    # combine embeddings
    x = tf.keras.layers.Concatenate(axis=-1)([emb_fft, emb_mel])
    output = SCAdaCos(n_classes=num_classes, n_subclusters=n_subclusters)([x, y, label_input])
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
    eps = 1e-12
    for label, category in enumerate(categories):
        print(category)
        for dict in dicts:
            for count, file in tqdm(enumerate(os.listdir(dict + category + "/train")),
                                    total=len(os.listdir(dict + category + "/train"))):
                file_path = dict + category + "/train/" + file
                wav, fs = sf.read(file_path)
                raw = librosa.core.to_mono(wav.transpose()).transpose()[:10 * target_sr]
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
if os.path.isfile(str(target_sr) + '_eval_raw.npy'):
    eval_raw = np.load(str(target_sr) + '_eval_raw.npy')
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
    eps = 1e-12
    for label, category in enumerate(categories):
        print(category)
        for count, file in tqdm(enumerate(os.listdir("./dev_data/" + category + "/test")),
                                total=len(os.listdir("./dev_data/" + category + "/test"))):
            file_path = "./dev_data/" + category + "/test/" + file
            wav, fs = sf.read(file_path)
            raw = librosa.core.to_mono(wav.transpose()).transpose()[:10 * target_sr]
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
        for count, file in tqdm(enumerate(os.listdir("./eval_data/" + category + "/test")),
                                total=len(os.listdir("./eval_data/" + category + "/test"))):
            file_path = "./eval_data/" + category + "/test/" + file
            wav, fs = sf.read(file_path)
            raw = librosa.core.to_mono(wav.transpose()).transpose()[:10 * target_sr]
            test_raw.append(raw)
            test_ids.append(category + '_' + file.split('_')[1])
            test_files.append(file_path)
    # reshape arrays and store
    test_ids = np.array(test_ids)
    test_files = np.array(test_files)
    test_raw = np.expand_dims(np.array(test_raw, dtype=np.float32), axis=-1)
    np.save('test_ids.npy', test_ids)
    np.save('test_files.npy', test_files)
    np.save(str(target_sr) + '_test_raw.npy', test_raw)

# encode ids as labels
le_4train = LabelEncoder()
train_ids_4train = np.array(['_'.join([train_ids[k], train_atts[k]]) for k in np.arange(train_ids.shape[0])])
eval_ids_4train = np.array(['_'.join([eval_ids[k], eval_atts[k]]) for k in np.arange(eval_ids.shape[0])])
le_4train.fit(np.concatenate([train_ids_4train, eval_ids_4train], axis=0))
num_classes_4train = len(np.unique(np.concatenate([train_ids_4train, eval_ids_4train], axis=0)))
train_labels_4train = le_4train.transform(train_ids_4train)
eval_labels_4train = le_4train.transform(eval_ids_4train)

le = LabelEncoder()
train_labels = le.fit_transform(train_ids)
eval_labels = le.transform(eval_ids)
test_labels = le.transform(test_ids)
num_classes = len(np.unique(train_labels))

# distinguish between normal and anomalous samples on development set
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

# training parameters
batch_size = 64
batch_size_test = 64
epochs = 100
aeons = 4
alpha = 1
n_subclusters = 16
ensemble_size = 10

# prepare scores and domain info
pred_eval = np.zeros((eval_raw.shape[0], np.unique(train_labels).shape[0]))
pred_unknown = np.zeros((unknown_raw.shape[0], np.unique(train_labels).shape[0]))
pred_test = np.zeros((test_raw.shape[0], np.unique(train_labels).shape[0]))
pred_train = np.zeros((train_labels.shape[0], np.unique(train_labels).shape[0]))
source_train = np.array([file.split('_')[3] == 'source' for file in train_files.tolist()])
source_eval = np.array([file.split('_')[3] == 'source' for file in eval_files.tolist()])
source_unknown = np.array([file.split('_')[3] == 'source' for file in unknown_files.tolist()])

for k_ensemble in np.arange(10):
    y_train_cat = keras.utils.np_utils.to_categorical(train_labels, num_classes=num_classes)
    y_eval_cat = keras.utils.np_utils.to_categorical(eval_labels, num_classes=num_classes)
    y_unknown_cat = keras.utils.np_utils.to_categorical(unknown_labels, num_classes=num_classes)
    y_test_cat = keras.utils.np_utils.to_categorical(test_labels, num_classes=num_classes)

    y_train_cat_4train = keras.utils.np_utils.to_categorical(train_labels_4train, num_classes=num_classes_4train)
    y_eval_cat_4train = keras.utils.np_utils.to_categorical(eval_labels_4train, num_classes=num_classes_4train)
    y_unknown_cat_4train = keras.utils.np_utils.to_categorical(unknown_labels_4train, num_classes=num_classes_4train)

    # compile model
    data_input, label_input, loss_output = model_emb_cnn(num_classes=num_classes_4train,
                                                             raw_dim=eval_raw.shape[1], n_subclusters=n_subclusters)
    model = tf.keras.Model(inputs=[data_input, label_input], outputs=[loss_output])
    model.compile(loss=[mixupLoss], optimizer=tf.keras.optimizers.Adam())
    #print(model.summary())
    for k in np.arange(aeons):
        print('ensemble iteration: ' + str(k_ensemble+1))
        print('aeon: ' + str(k+1))
        # fit model
        weight_path = 'wts_' + str(k+1) + 'k_' + str(target_sr) + '_' + str(k_ensemble+1) + '.h5'
        if not os.path.isfile(weight_path):
            model.fit(
                [train_raw, y_train_cat_4train], y_train_cat_4train, verbose=1,
                batch_size=batch_size, epochs=epochs,
                validation_data=([eval_raw, y_eval_cat_4train], y_eval_cat_4train))
            model.save(weight_path)
        else:
            model = tf.keras.models.load_model(weight_path,
                                               custom_objects={'MixupLayer': MixupLayer, 'mixupLoss': mixupLoss,
                                                               'SCAdaCos': SCAdaCos,
                                                               'LogMelSpectrogram': LogMelSpectrogram})

        # extract embeddings
        emb_model = tf.keras.Model(model.input, model.layers[-3].output)
        eval_embs = emb_model.predict([eval_raw, np.zeros((eval_raw.shape[0], num_classes_4train))], batch_size=batch_size)
        train_embs = emb_model.predict([train_raw, np.zeros((train_raw.shape[0], num_classes_4train))], batch_size=batch_size)
        unknown_embs = emb_model.predict([unknown_raw, np.zeros((unknown_raw.shape[0], num_classes_4train))], batch_size=batch_size)
        test_embs = emb_model.predict([test_raw, np.zeros((test_raw.shape[0], num_classes_4train))], batch_size=batch_size)

        # length normalization
        x_train_ln = length_norm(train_embs)
        x_eval_ln = length_norm(eval_embs)
        x_test_ln = length_norm(test_embs)
        x_unknown_ln = length_norm(unknown_embs)

        # compute ASD scores
        n_subclusters_gmm = n_subclusters
        for j, lab in tqdm(enumerate(np.unique(train_labels)), total=len(np.unique(train_labels))):
            if np.sum(train_labels == lab) > 1:
                # domain mixup
                x_train_ln_cp = np.copy(x_train_ln[source_train * (train_labels == lab)])
                rand_target_idcs = np.random.randint(0, np.sum(~source_train * (train_labels == lab)),
                                                     np.sum(source_train * (train_labels == lab)))
                rand_mix_coeffs = 0.5
                x_train_ln_cp *= rand_mix_coeffs
                x_train_ln_cp += (1 - rand_mix_coeffs) * x_train_ln[~source_train * (train_labels == lab)][
                    rand_target_idcs]
                # train and evaluate GMMs
                if np.sum(train_labels == lab) >= n_subclusters_gmm:
                    clf1 = GaussianMixture(n_components=n_subclusters_gmm, covariance_type='full',
                                           reg_covar=1e-3).fit(
                        np.concatenate([x_train_ln_cp, x_train_ln[train_labels == lab]], axis=0))
                else:
                    clf1 = GaussianMixture(n_components=np.sum(train_labels == lab), covariance_type='full',
                                           reg_covar=1e-3).fit(
                        np.concatenate([x_train_ln_cp, x_train_ln[train_labels == lab]], axis=0))
                pred_train[train_labels==lab, j] += -clf1.score_samples(x_train_ln[train_labels==lab])
                if np.sum(eval_labels == lab) > 0:
                    pred_eval[eval_labels == lab, j] += -clf1.score_samples(x_eval_ln[eval_labels == lab])
                    pred_unknown[unknown_labels == lab, j] += -clf1.score_samples(x_unknown_ln[unknown_labels == lab])

                if np.sum(test_labels == lab) > 0:
                    pred_test[test_labels == lab, j] += -clf1.score_samples(x_test_ln[test_labels == lab])

        # print results for development set
        print('#######################################################################################################')
        print('DEVELOPMENT SET')
        print('#######################################################################################################')
        aucs = []
        p_aucs = []
        for j, cat in enumerate(np.unique(eval_ids)):
            y_pred = np.concatenate([pred_eval[eval_labels == le.transform([cat]), le.transform([cat])],
                                     pred_unknown[unknown_labels == le.transform([cat]), le.transform([cat])]],
                                     axis=0)
            y_true = np.concatenate([np.zeros(np.sum(eval_labels == le.transform([cat]))),
                                     np.ones(np.sum(unknown_labels == le.transform([cat])))], axis=0)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
            p_aucs.append(p_auc)
            print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))

            source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])],
                                         source_unknown[unknown_labels == le.transform([cat])]], axis=0)
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
            mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
            print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
        print('####################')
        mean_auc = hmean(aucs)
        print('mean AUC: ' + str(mean_auc * 100))
        mean_p_auc = hmean(p_aucs)
        print('mean pAUC: ' + str(mean_p_auc * 100))

        """
        # print results for eval set
        print('#######################################################################################################')
        print('EVALUATION SET')
        print('#######################################################################################################')
        aucs = []
        p_aucs = []
        for j, cat in enumerate(np.unique(test_ids)):
            y_pred = pred_test[test_labels == le.transform([cat]), le.transform([cat])]
            y_true = np.array(pd.read_csv(
                './dcase2022_evaluator-main/ground_truth_data/ground_truth_' + cat.split('_')[0] + '_section_' + cat.split('_')[1] + '_test.csv', header=None).iloc[:, 1] == 1)
            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)
            p_auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
            p_aucs.append(p_auc)
            print('AUC for category ' + str(cat) + ': ' + str(auc * 100))
            print('pAUC for category ' + str(cat) + ': ' + str(p_auc * 100))

            source_all = np.concatenate([source_eval[eval_labels == le.transform([cat])],
                                         source_unknown[unknown_labels == le.transform([cat])]], axis=0)
            source_all = np.array(pd.read_csv(
                './dcase2022_evaluator-main/ground_truth_domain/ground_truth_' + cat.split('_')[0] + '_section_' + cat.split('_')[1] + '_test.csv', header=None).iloc[:, 1] == 0)
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
            mean_p_auc = hmean(p_aucs[np.array([eval_id.split('_')[0] for eval_id in np.unique(eval_ids)]) == cat])
            print('mean of AUC and pAUC for category ' + str(cat) + ': ' + str((mean_p_auc + mean_auc) * 50))
        print('####################')
        mean_auc = hmean(aucs)
        print('mean AUC: ' + str(mean_auc * 100))
        mean_p_auc = hmean(p_aucs)
        print('mean pAUC: ' + str(mean_p_auc * 100))
        """

# create challenge submission files
print('creating submission files')
sub_path = './submission'
if not os.path.exists(sub_path):
    os.makedirs(sub_path)
for j, cat in enumerate(np.unique(test_ids)):
    # anomaly scores
    file_idx = test_labels == le.transform([cat])
    results_an = pd.DataFrame()
    results_an['output1'], results_an['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                    [str(s) for s in pred_test[file_idx, le.transform([cat])]]]
    results_an.to_csv(sub_path + '/anomaly_score_' + cat.split('_')[0] + '_section_' + cat.split('_')[-1] + '_test.csv',
                      encoding='utf-8', index=False, header=False)

    # decision results
    train_scores = pred_train[train_labels == le.transform([cat]), le.transform([cat])]
    threshold = np.percentile(train_scores, q=90)
    decisions = pred_test[file_idx, le.transform([cat])] > threshold
    results_dec = pd.DataFrame()
    results_dec['output1'], results_dec['output2'] = [[f.split('/')[-1] for f in test_files[file_idx]],
                                                      [str(int(s)) for s in decisions]]
    results_dec.to_csv(sub_path + '/decision_result_' + cat.split('_')[0] + '_section_' + cat.split('_')[-1] + '_test.csv',
                       encoding='utf-8', index=False, header=False)
print('####################')
print('>>>> finished! <<<<<')
print('####################')
