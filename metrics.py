import math
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K


class AdaCos_RBF(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, regularizer=None, **kwargs):
        super(AdaCos_RBF, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s_init = math.sqrt(2) * math.log(n_classes - 1)
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(AdaCos_RBF, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)
        self.s = self.add_weight(shape=(),
                                  initializer=tf.keras.initializers.Constant(self.s_init),
                                  trainable=False,
                                  aggregation=tf.VariableAggregation.MEAN)
        # replace with covariance matrix instead of single gamma?
        self.gamma = self.add_weight(name='gamma',
                                  shape=(1, self.n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self.regularizer)

    def call(self, inputs, training=None):
        x, y = inputs
        # normalize feature
        xn = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = xn @ W  # same as cos theta
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        if training:
            B_avg = tf.where(y < 1, tf.exp(self.s * logits), tf.zeros_like(logits))
            B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1), name='B_avg')
            theta_class = tf.gather(theta, tf.cast(y, tf.int32), name='theta_class')
            theta_med = tfp.stats.percentile(theta_class, q=50)  # computes median
            self.s.assign(
                tf.math.log(B_avg) /
                tf.math.cos(tf.minimum(math.pi / 4, theta_med)))
        logits *= self.s
        # RBF
        diff = K.expand_dims(x, axis=-1) - W
        l2 = K.sum(K.pow(diff, 2), axis=1)
        l = K.exp(-K.pow(self.gamma, 2) * l2)  # enforce a positive gamma
        # logits *= l  # multiply with softmax before or after softmax?
        out = (tf.keras.activations.softmax(logits)+l)/2
        # product or mean? mean is more independent because when taking the mean both must be reduced, not when using a product
        #return [out, self.gamma]  # minimize gamma to have very small support?
        # binary or categorical x-entropy?
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = {
            'n_classes': self.n_classes,
            'regularizer': self.regularizer
        }
        base_config = super(AdaCos_RBF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TypeConverter(tf.keras.layers.Layer):
    def __init__(self, type_mat, regularizer=None, **kwargs):
        super(TypeConverter, self).__init__(**kwargs)
        self.type_mat = type_mat
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(TypeConverter, self).build(input_shape[0])
        self.W = self.add_weight(shape=(41, 6),
                                 initializer=tf.keras.initializers.Constant(self.type_mat),
                                 trainable=False,
                                 regularizer=self.regularizer)

    def call(self, inputs, training=None):
        return tf.python.keras.layers.ops.core.dense(inputs, self.type_mat)

    def compute_output_shape(self, input_shape):
        return (None, self.type_mat.shape[1])

    def get_config(self):
        config = {
            'type_mat': self.type_mat,
            'regularizer': self.regularizer
        }
        base_config = super(TypeConverter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaCos(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, n_subclusters=1, regularizer=None, **kwargs):
        super(AdaCos, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.n_subclusters = n_subclusters
        self.s_init = math.sqrt(2) * math.log(n_classes*n_subclusters - 1)
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(AdaCos, self).build(input_shape[0])
        self.W = self.add_weight(name='W_AdaCos' + str(self.n_classes) + '_' + str(self.n_subclusters),
                                 shape=(input_shape[0][-1], self.n_classes*self.n_subclusters),
                                 initializer='glorot_uniform',
                                 trainable=False,
                                 regularizer=self.regularizer)
        self.s = self.add_weight(name='s' + str(self.n_classes) + '_' + str(self.n_subclusters),
                                 shape=(),
                                  initializer=tf.keras.initializers.Constant(self.s_init),
                                  trainable=False,
                                  aggregation=tf.VariableAggregation.MEAN)

    def call(self, inputs, training=None):
        x, y1, y2 = inputs
        y1_orig = y1
        y1 = tf.repeat(y1, repeats=self.n_subclusters, axis=-1)
        y2 = tf.repeat(y2, repeats=self.n_subclusters, axis=-1)
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W  # same as cos theta
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))

        if training:
            max_s_logits = tf.reduce_max(self.s * logits)
            # B_avg = tf.where(y1 < 1, tf.exp(self.s * logits), tf.zeros_like(logits))  # non-corresponding classes (after mixup)
            #B_avg = tf.where(y1 < K.epsilon(), tf.exp(self.s * logits - max_s_logits),
            #                 tf.zeros_like(logits))  # non-corresponding classes (after mixup)
            B_avg = tf.exp(self.s*logits-max_s_logits)
            B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1))
            theta_class = tf.reduce_sum(y1 * theta, axis=1) * tf.math.count_nonzero(y1_orig, axis=1, dtype=tf.dtypes.float32)  # take mix-upped angle of mix-upped classes
            theta_med = tfp.stats.percentile(theta_class, q=50)  # computes median
            self.s.assign(
                (max_s_logits + tf.math.log(B_avg)) /
                # (tf.math.log(B_avg)) /
                tf.math.cos(tf.minimum(math.pi / 4, theta_med)) + K.epsilon())
        logits *= self.s
        """
        target_logits = tf.cos(theta + 0.05)
        logits = logits * (1 - y1) + target_logits * y1
        logits *= 32#self.s
        """
        #logits = tf.reshape(logits, (-1, self.n_classes, self.n_subclusters))
        #logits = tf.math.reduce_max(logits, axis=2)
        out = tf.keras.activations.softmax(logits)
        out = tf.reshape(out, (-1, self.n_classes, self.n_subclusters))
        out = tf.math.reduce_sum(out, axis=2)
        #out = tf.math.reduce_max(out, axis=2)
        #out = tf.keras.activations.softmax(out)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = {
            'n_classes': self.n_classes,
            'regularizer': self.regularizer,
            'n_subclusters': self.n_subclusters
        }
        base_config = super(AdaCos, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IntraLoss(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, regularizer=None, **kwargs):
        super(IntraLoss, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s#math.sqrt(2) * math.log(n_classes - 1)
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(IntraLoss, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta)
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)*tf.exp(-theta/math.pi)  # because of logarithm that is applied when using categorical crossentropy

        return out

    def get_config(self):
        config = {
            'n_classes': self.n_classes,
            's': self.s,
            'regularizer': self.regularizer
        }
        base_config = super(IntraLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def get_config(self):
        config = {
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        }
        base_config = super(ArcFace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SphereFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=1.35, regularizer=None, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(SphereFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(self.m * theta)
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def get_config(self):
        config = {
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        }
        base_config = super(SphereFace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CosFace(tf.keras.layers.Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        target_logits = logits - self.m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def get_config(self):
        config = {
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        }
        base_config = super(CosFace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
