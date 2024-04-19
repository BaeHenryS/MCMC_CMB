""""
Bayesian neural network based on Dropout(Gal et.al., 2015).
Code for extracting the cosmological CMB parameters.
architecture implemented in sonnet: https://sonnet.readthedocs.io/en/latest/
Main Author:
Hector Javier Hortua
orjuelajavier@gmail.com

"""


import json
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sonnet as snt
from collections import defaultdict
import tensorflow_probability as tfp
import math
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import warnings
import matplotlib.cbook
from scipy.stats import chi2
#from concrete_dropout import concrete_dropout
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"




import tensorflow as tf
import numpy as np

from tensorflow.python.layers import base
from tensorflow.python.layers import utils

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops


class ConcreteDropout(base.Layer):
    """Concrete Dropout layer class from https://arxiv.org/abs/1705.07832.

    "Concrete Dropout" Yarin Gal, Jiri Hron, Alex Kendall

    Arguments:
        weight_regularizer:
            Positive float, satisfying $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
            (inverse observation noise), and N the number of instances
            in the dataset.
        dropout_regularizer:
            Positive float, satisfying $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
            N the number of instances in the dataset.
            The factor of two should be ignored for cross-entropy loss,
            and used only for the eucledian loss.
        init_min:
            Minimum value for the randomly initialized dropout rate, in [0, 1].
        init_min:
            Maximum value for the randomly initialized dropout rate, in [0, 1],
            with init_min <= init_max.
        name:
            String, name of the layer.
        reuse:
            Boolean, whether to reuse the weights of a previous layer
            by the same name.
    """

    def __init__(self, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, name=None, reuse=False,
                 training=True, **kwargs):

        super(ConcreteDropout, self).__init__(name=name, _reuse=reuse,
                                              **kwargs)
        assert init_min <= init_max, \
            'init_min must be lower or equal to init_max.'

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))
        self.training = training
        self.reuse = reuse

    def get_kernel_regularizer(self):
        def kernel_regularizer(weight):
            if self.reuse:
                return None
            return self.weight_regularizer * tf.reduce_sum(tf.square(weight)) \
                / (1. - self.p)
        return kernel_regularizer

    def apply_dropout_regularizer(self, inputs):
        with tf.name_scope('dropout_regularizer'):
            input_dim = tf.cast(tf.reduce_prod(tf.shape(inputs)[1:]),
                                dtype=tf.float32)
            dropout_regularizer = self.p * tf.log(self.p)
            dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 dropout_regularizer)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        self.input_spec = base.InputSpec(shape=input_shape)

        self.p_logit = self.add_variable(name='p_logit',
                                         shape=[],
                                         initializer=tf.random_uniform_initializer(
                                             self.init_min,
                                             self.init_max),
                                         dtype=tf.float32,
                                         trainable=True)
        self.p = tf.nn.sigmoid(self.p_logit, name='dropout_rate')
        tf.add_to_collection('DROPOUT_RATES', self.p)

        self.built = True

    def concrete_dropout(self, x):
        eps = 1e-7
        temp = 0.1

        with tf.name_scope('dropout_mask'):
            unif_noise = tf.random_uniform(shape=tf.shape(x))
            drop_prob = (
                tf.log(self.p + eps)
                - tf.log(1. - self.p + eps)
                + tf.log(unif_noise + eps)
                - tf.log(1. - unif_noise + eps)
            )
            drop_prob = tf.nn.sigmoid(drop_prob / temp)

        with tf.name_scope('drop'):
            random_tensor = 1. - drop_prob
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob

        return x

    def call(self, inputs, training=True):
        def dropped_inputs():
            return self.concrete_dropout(inputs)
        if not self.reuse:
            self.apply_dropout_regularizer(inputs)
        return utils.smart_cond(training,
                                dropped_inputs,
                                lambda: array_ops.identity(inputs))


def concrete_dropout(inputs,
                     trainable=True,
                     weight_regularizer=1e-6,
                     dropout_regularizer=1e-5,
                     init_min=0.1, init_max=0.1,
                     training=True,
                     name=None,
                     reuse=False,
                     **kwargs):

    """Functional interface for Concrete Dropout.

    "Concrete Dropout" Yarin Gal, Jiri Hron, Alex Kendall
    from https://arxiv.org/abs/1705.07832.

    Arguments:
        weight_regularizer:
            Positive float, satisfying $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
            (inverse observation noise), and N the number of instances
            in the dataset.
        dropout_regularizer:
            Positive float, satisfying $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
            N the number of instances in the dataset.
            The factor of two should be ignored for cross-entropy loss,
            and used only for the eucledian loss.
        init_min:
            Minimum value for the randomly initialized dropout rate, in [0, 1].
        init_min:
            Maximum value for the randomly initialized dropout rate, in [0, 1],
            with init_min <= init_max.
        name:
            String, name of the layer.
        reuse:
            Boolean, whether to reuse the weights of a previous layer
            by the same name.

    Returns:
        Tupple containing:
            - the output of the dropout layer;
            - the kernel regularizer function for the subsequent
              convolutional layer.
    """

    layer = ConcreteDropout(weight_regularizer=weight_regularizer,
                            dropout_regularizer=dropout_regularizer,
                            init_min=init_min, init_max=init_max,
                            training=training,
                            trainable=trainable,
                            name=name,
                            reuse=reuse)
    return layer.apply(inputs, training=training), \
        layer.get_kernel_regularizer()




""""
Load dataset generated by CMB data generator.
Dataset provided in Pandas Dataframe
"""
image_File = './ssd_data/CMB/old/omega_b_omega_cdm_A_s-uniform-pppr_rot0-pppr_eq1-pppr_lat1-aug0-cod_lt+patch_s10+pic_s256'
json_filename =image_File+  "/params.json"
list_params=['omega_b','omega_cdm','A_s']
json_filename =image_File+  "/params.json"
csv_filename = image_File + '/labels_file.csv'
label_df = pd.read_csv(csv_filename, sep="\t")

def compute_min_max_data():
    all_max = []
    all_min = []
    csv_filename = image_File + '/labels_file.csv'
    label_df = pd.read_csv(csv_filename, sep="\t")
    filenames = label_df['filename']

    for filename in filenames:
        patch1 = np.load(image_File + "/" + filename)

        all_min.append(np.min(patch1))
        all_max.append(np.max(patch1))

    data_min = np.min(all_min)
    data_max = np.max(all_max)
    return data_min, data_max

data_min, data_max = compute_min_max_data()

labels_norm = label_df[list_params].values.astype(np.float32)
max_train = np.max(labels_norm,axis=0)
min_train = np.min(labels_norm,axis=0)

from sklearn.model_selection import train_test_split

# Load the labels file
csv_filename = image_File + '/labels_file.csv'
label_df = pd.read_csv(csv_filename, sep="\t")

# Split the data into training and the rest (70% training, 30% the rest)
train_df, rest_df = train_test_split(label_df, test_size=0.3, random_state=42)

# Split the rest into test and validation (50% test, 50% validation)
test_df, validation_df = train_test_split(rest_df, test_size=0.5, random_state=42)

# Save the dataframes to CSV files
train_df.to_csv(image_File + '/Train_CMB_data.csv', sep="\t", index=False)
test_df.to_csv(image_File + '/Test_CMB_data.csv', sep="\t", index=False)
validation_df.to_csv(image_File + '/Validation_CMB_data.csv', sep="\t", index=False)


csv_filename_Train = image_File + '/Train_CMB_data.csv'
label_df_Train = pd.read_csv(csv_filename_Train, sep="\t")

csv_filename_Test = image_File + '/Test_CMB_data.csv'
label_df_Test = pd.read_csv(csv_filename_Test, sep="\t")

csv_filename_Validation = image_File + '/Validation_CMB_data.csv'
label_df_Validation = pd.read_csv(csv_filename_Validation, sep="\t")


shuffle_data_Train= label_df_Train.sample(frac=1)
label_df_Train=shuffle_data_Train

shuffle_data_Test= label_df_Test.sample(frac=1)
label_df_Test=shuffle_data_Test

shuffle_data_Validation= label_df_Validation.sample(frac=1)
label_df_Validation=shuffle_data_Validation

label_df_Test = pd.concat([label_df_Test, label_df_Validation], ignore_index=True)


#train
filenames_training = label_df_Train['filename'].values

labels_training = label_df_Train[list_params].values.astype(np.float32)


labels_training = 2*(labels_training-min_train)/(max_train-min_train)-1


#testing
filenames_testing = label_df_Test['filename'].values
labels_testing = label_df_Test[list_params].values.astype(np.float32)

labels_testing =  2*(labels_testing-min_train)/(max_train-min_train)-1

size_image = np.load(image_File+'/'+filenames_testing[0]).shape[1]
size_label = labels_testing.shape[1]

#Validation
filenames_validation = label_df_Validation['filename'].values
labels_validation = label_df_Validation[list_params].values.astype(np.float32)

labels_validation =  2*(labels_validation-min_train)/(max_train-min_train)-1

size_label_Validation = labels_validation.shape[1]


dim_train=len(labels_training)
dim_test=len(labels_testing)
dim_validation=len(labels_validation)


def parse_function(filename, label):
    image = np.load(image_File+'/'+filename.decode())
    image = 2*(image-data_min)/(data_max-data_min)-1
    image = np.expand_dims(image, axis=-1)

    return image.astype(np.float32), label

batch_size = 32

#Training
dataset_training = tf.data.Dataset.from_tensor_slices((filenames_training, labels_training))
dataset_training = dataset_training.shuffle(len(filenames_training))
dataset_training = dataset_training.map(
    lambda filename, label: tuple(
    tf.numpy_function(
        parse_function, [filename, label], [tf.float32, label.dtype])
    ), num_parallel_calls=4)

#dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset_training = dataset_training.batch(batch_size)
dataset_training = dataset_training.prefetch(1)

iterator_training = iter(dataset_training)
X_image_train, Y_labels_train = next(iterator_training)
X_image_train.set_shape([None, size_image, size_image, 1])
Y_labels_train.set_shape([None, size_label])

#Testing
dataset_testing = tf.data.Dataset.from_tensor_slices((filenames_testing, labels_testing))
dataset_testing = dataset_testing.shuffle(len(filenames_testing))
dataset_testing = dataset_testing.map(
     lambda filename, label: tuple(
     tf.numpy_function(
         parse_function, [filename, label], [tf.float32, label.dtype])
     )
    , num_parallel_calls=4)
#dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset_testing = dataset_testing.batch(batch_size)
dataset_testing = dataset_testing.prefetch(1)

# Create a Python iterator object from the dataset
iterator_testing = iter(dataset_testing)

# Use the iterator to get the next batch of data
X_image_test, Y_labels_test = next(iterator_testing)

# Set the shapes of the data
X_image_test.set_shape([None, size_image, size_image, 1])
Y_labels_test.set_shape([None, size_label])


#Validation
dataset_validation = tf.data.Dataset.from_tensor_slices((filenames_validation, labels_validation))
dataset_validation = dataset_validation.shuffle(len(filenames_validation))
dataset_validation = dataset_validation.map(
     lambda filename, label: tuple(
     tf.numpy_function(
         parse_function, [filename, label], [tf.float32, label.dtype])
     )
    , num_parallel_calls=4)
#dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset_validation = dataset_validation.batch(batch_size)
dataset_validation = dataset_validation.prefetch(1)

# Create a Python iterator object from the dataset
iterator_validation = iter(dataset_validation)

# Use the iterator to get the next batch of data
X_image_validation, Y_labels_validation = next(iterator_validation)

# Set the shapes of the data
X_image_validation.set_shape([None, size_image, size_image, 1])
Y_labels_validation.set_shape([None, size_label])

tf.keras.backend.clear_session()

"""""
Architecture, dropout is applied
after each layer
"""
class Network2(snt.Module):
    def __init__(self, num_classes, drop_out_rate, name=None):
        super(Network2, self).__init__(name=name)
        self._num_classes = num_classes
        self._output_channels = [16, 16, 32, 32, 32]
        self._num_layers = len(self._output_channels)
        self._kernel_shapes = [[3,3]]*(self._num_layers)
        self._strides = [4,1,1,1,1]
        self._paddings = ['VALID']+['SAME']*(self._num_layers-1)
        self._rate = drop_out_rate
        self._initializer = {'w':tf.initializers.GlorotUniform(),
                             'b':tf.constant_initializer(10)}
        self._layers = [snt.Conv2D(name="conv2d{}".format(i).replace('/', '_'),
                                   output_channels=self._output_channels[i],
                                   kernel_shape=self._kernel_shapes[i],
                                   stride=self._strides[i],
                                   padding=self._paddings[i],
                                   w_init=self._initializer['w'],
                                   b_init=self._initializer['b'],
                                   with_bias=True) for i in range(self._num_layers)]
        self._final_layer = snt.Conv2D(name="conv2dl1".replace('/', '_'),
                                       output_channels=256,
                                       kernel_shape=[2,2],
                                       stride=1,
                                       padding='VALID',
                                       w_init=self._initializer['w'],
                                       b_init=self._initializer['b'],
                                       with_bias=True)
        self._linear = snt.Linear(self._num_classes)

    def __call__(self, inputs, is_training=None):
        test_local_stats = is_training
        net = inputs
        for i, layer in enumerate(self._layers):
            net = layer(net)
            net = tf.keras.layers.Dropout(self._rate)(net, training=is_training)
            bn = snt.BatchNorm(create_scale=True, create_offset=True, name="batch_norm".format(i))
            net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
            net = tf.nn.relu(net)
            if i==0 or i==1 or i==4:
                net = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="SAME")(net)
        net = self._final_layer(net)
        net = tf.keras.layers.Dropout(self._rate)(net, training=is_training)
        bn = snt.BatchNorm(create_scale=True, create_offset=True, name="batch_norm_ff")
        net = bn(net, is_training=is_training, test_local_stats=test_local_stats)
        net = tf.nn.relu(net)
        net = tf.keras.layers.Flatten()(net)
        logits = self._linear(net)
        return logits
    

_drop_out_rate=0.1
size_label=len(list_params)
size_nn= int(size_label+ size_label*(size_label+1)/2)
model_1 = Network2(num_classes=size_nn,drop_out_rate= _drop_out_rate, name="network2")

Y_predicted = model_1(X_image_train, is_training=True)
#regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in model_1.trainable_variables])
snt.allow_empty_variables(model_1)
test_predictions = model_1(X_image_test, is_training=False)
tolerance=10e-4

def make_dist(prediction_nn, z_size=size_label):
    """"
    The network's output are located in the covariance matrix
    semipositive definite

    """
    net = prediction_nn
    covariance_weights = net[:, z_size:]
    lower_triangle = tfp.math.fill_triangular(covariance_weights)
    diag = tf.linalg.diag_part(lower_triangle)
    diag_positive = tf.nn.softplus(diag)+tolerance
    covariance_matrix = lower_triangle - tf.linalg.diag(diag) + tf.linalg.diag(diag_positive)

    distri=tfp.distributions.MultivariateNormalTriL(
        loc=net[:,:z_size], scale_tril=covariance_matrix)
    distri_mean = distri.mean()
    distri_covariance =distri.covariance()
    distri_variance =distri.variance()
    return distri,distri_mean,distri_variance,distri_covariance

"""""
minimize negative_log_likelihood
"""


step = tf.Variable(0, trainable=False, dtype=tf.int32)

def learning_rate_fn(step):
    initial_learning_rate = 10e-4
    decay_steps = 250
    decay_rate = 1/math.e

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)

    return 10e-6 + lr_schedule(step)
    
Distr_train,Distr_mean_train,Distr_var_train,Distr_covar_train = make_dist(Y_predicted)
Distr_test,Distr_mean_test,Distr_var_test,Distr_covar_test   = make_dist(test_predictions)
negative_log_likelihood_train =  -tf.reduce_mean(Distr_train.log_prob(Y_labels_train))
negative_log_likelihood_test =  -tf.reduce_mean(Distr_test.log_prob(Y_labels_test))

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

# # Ensure the model is built by calling it with some input data
# model_1.build(X_image_train.shape)


with tf.GradientTape() as tape:
    Y_predicted = model_1(X_image_train, is_training=True)
    Distr_train, _, _, _ = make_dist(Y_predicted)
    loss = -tf.reduce_mean(Distr_train.log_prob(Y_labels_train))
    print(loss)

gradients = tape.gradient(loss, [tf.Variable(var.numpy(), name=var.name.replace('/', '_')) for var in model_1.trainable_variables])

print("Gradients: ", gradients)

if not gradients:
    raise ValueError("No gradients were computed. Check the model and the loss function.")

train_op = optimizer.apply_gradients(zip(gradients, model_1.trainable_variables))

X_feeded_image_test = tf.keras.Input(shape=[size_image,size_image,1])
Distr_test_feed, Distr_mean_test_feed, Distr_var_test_feed,Distr_covar_test_feed = make_dist(model_1(X_feeded_image_test, training=False))
Distr_test_feed_sample = Distr_test_feed.sample()
Directory_plot_name='Plots_Bayes/'
suffix_='BNNDropout_{}'.format(_drop_out_rate)
Directory_data_name='data_Bayes/'
plots_name=Directory_plot_name+suffix_
data_name=Directory_data_name+suffix_

def coef_deter(predic,real):
    mean=np.mean(real,axis=0)
    numerator=np.sum(np.square(predic -real),axis=0)
    denominator=np.sum(np.square(real -mean),axis=0)
    return 1.-(numerator/denominator)

def unnormalized(valu):
        unormal= valu*(max_train-min_train)+min_train
        num = 0.0002
        scalar = 1e8
        b = tuple({i for i in range(unormal.shape[1]) if i > num})
        unormal[:, b] *= scalar
        return unormal

def CI(predictions, rea_valu, variance_s, covariance_s):
    """"
    Function used for computing the coverage probability
    for confidence invervals 1,2,3 \sigma
    """

    batch_size = rea_valu.shape[0]
    mean_pred =  np.mean(predictions, axis=2)
    var_pred = np.var(predictions, axis=2)
    cov_pred = np.array(list(map(lambda x: np.cov(x), predictions)))
    mean_var = np.mean(variance_s, axis=2)
    mean_covar = np.mean(covariance_s, axis=3)
    total_variance = var_pred + mean_var
    total_covariance = cov_pred + mean_covar
    total_std = np.sqrt(total_variance)
    summa_68_95_99 = general_ellip_counts(total_covariance, mean_pred, rea_valu)

    return summa_68_95_99, batch_size,\
           total_std, cov_pred, mean_covar, total_covariance, rea_valu, mean_pred

def general_ellip_counts(covariance, mean, real_values):
    Inverse_covariance = np.linalg.inv(covariance)
    Ellip_eq = np.einsum('nl,nlm,mn->n', (real_values - mean), Inverse_covariance, (real_values - mean).T)
    ppf_run=[0.68,0.955,0.997]
    summa_T=[0,0,0]
    rv = chi2(df=mean.shape[1])
    for ix, ppf in enumerate(ppf_run):
        square_norm = rv.ppf(ppf)
        values = Ellip_eq / square_norm
        for ids, inst in enumerate(values):
            if inst <= 1:
                summa_T[ix] += 1
            else:
                pass
    return summa_T



num_batches = dim_train//batch_size
num_batches_test = dim_test//batch_size
count_posterior =2500
epochs=100
with tf.Session(config=config) as session:
    session.run(init)
    loss=[]
    num_elements =[]
    num_elements1 =[]

    for i in range(1, epochs):
        for X_image_train, Y_labels_train in dataset_training:
            _, loss_val, predic_val_train, rea_valu_train, std_train, lr = training_baseline_op(
                negative_log_likelihood_train,
                Distr_mean_train, Y_labels_train, Distr_var_train,
                learning_rate, step=i
            )
            val_coef = coef_deter(predic_val_train, rea_valu_train)
            print('train_log: {}  R^2: {} Epoch: {}'.format(loss_val ,val_coef, i))

        list_conf_68=0
        list_conf_95=0
        list_conf_99=0
        list_conf_T =0
        list_conf_68_95_99=[]

        loss_val_tests = negative_log_likelihood_test()
        with open(data_name+'loss.dat', 'a') as ft:
                ft.write("{} {} {} \n".format(i, loss_val, loss_val_tests))
        ft.close()
        number_test_images_running=100
        MCdrop=60
        if  i%MCdrop==0:
            for number_test_images in range(number_test_images_running):
                loss_val_test,predic_val,rea_valu,std,X_image_feed = negative_log_likelihood_test(
                                    Distr_mean_test, Y_labels_test, Distr_var_test,X_image_test
                                                                                )

                testpred_feeded_total =[]
                testpred_feeded_total_var=[]
                testpred_feeded_total_covar=[]
                testpred_feeded_total_sample=[]

                for _ in range(count_posterior):

                    means_feed, var_feed,covar_feed,sample_feed = Distr_mean_test_feed(
                                                                            Distr_var_test_feed,
                                                                            Distr_covar_test_feed,
                                                                            Distr_test_feed_sample,
                                                                            X_feeded_image_test=X_image_feed)


                    testpred_feeded_total.append(means_feed)
                    testpred_feeded_total_var.append(var_feed)
                    testpred_feeded_total_covar.append(covar_feed)
                    testpred_feeded_total_sample.append(sample_feed)

                testpred_feeded_stack = np.stack(testpred_feeded_total, axis=2)
                testpred_feeded_stack_var = np.stack(testpred_feeded_total_var, axis=2)
                testpred_feeded_stack_sample=np.stack(testpred_feeded_total_sample, axis=2)
                testpred_feeded_stack_covar = np.stack(testpred_feeded_total_covar, axis=3)
                coverage_v_68_95_99,coverage_T,\
                Total_std, cov_of_predictions, means_of_covar, Total_covariance, rea_valu, mean_pred = CI(
                                                                testpred_feeded_stack,
                                                                rea_valu,
                                                                testpred_feeded_stack_var,
                                                                testpred_feeded_stack_covar)

                list_conf_68 += coverage_v_68_95_99[0]
                list_conf_95 += coverage_v_68_95_99[1]
                list_conf_99 += coverage_v_68_95_99[2]
                list_conf_T  += coverage_T

            list_conf_68_95_99=[list_conf_68,list_conf_95,list_conf_99]

            np.save(data_name+'means_means_{}_{}'.format(number_test_images,i) , mean_pred)
            np.save(data_name+'covs_means_{}_{}'.format(number_test_images,i) , means_of_covar)
            np.save(data_name+'means_covs_{}_{}'.format(number_test_images,i) , cov_of_predictions)
            np.save(data_name+'chains_{}_{}'.format(number_test_images,i) , testpred_feeded_stack_sample)
            np.save(data_name+'real_value_{}_{}'.format(number_test_images,i) , rea_valu)
            np.save(data_name+'Total_std_{}_{}'.format(number_test_images,i) , Total_std)

            print('conf.68_95_99: {}    \
                count_T: {}'.format(list_conf_68_95_99,list_conf_T))
            print('test_log: {}  R^2:{} Epoch: {}'.format(loss_val_test,
                                                        coef_deter(predic_val,rea_valu),
                                                        i))
            fig_1 = plt.figure()
            colors=['blue','red','green','orange', 'darkviolet','purple']
            ecolors=['lightblue','lightcoral','lightgreen','yellow', 'violet','magenta']
            for m in range(len(list_params)):
                plt.errorbar(rea_valu[:,m], mean_pred[:,m], Total_std[:,m],fmt='o', color=colors[m],
                                    ecolor=ecolors[m], elinewidth=3, capsize=0, label=list_params[m])

                line_s1=np.arange(-0.01,1,0.01)
                plt.plot(line_s1,line_s1,'r-', alpha=0.1)
                plt.xlabel('True value')
                plt.ylabel('Predicted value')
                plt.legend()

            plt.savefig(plots_name+"plot_parameters_{}_{}.png".format(i,number_test_images))
            plt.close(fig_1)
            with open(data_name+'info_covs_{}.dat'.format(i), 'a') as ft1:
                ft1.write("{} {} {} {} {} {}\n".format(list_conf_68_95_99, list_conf_T,
                                                np.array(list_conf_68_95_99)/list_conf_T,max_train,min_train))
            ft1.close()