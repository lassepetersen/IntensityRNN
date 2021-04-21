import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from intensitymodel import IntensityRNN, LaggedSequence, create_dataset
import matplotlib.pyplot as plt

# Simulating renewal process data

N = 500

data_exp = []
data_lognorm = []
data_unif = []
for i in np.arange(N):
    T = 30

    tmp1 = np.random.exponential(1, int(T)).cumsum()
    tmp2 = 15
    data_exp.append([tmp2, [tmp1]])

    tmp1 = np.random.lognormal(3, 0.2, int(T)).cumsum()
    tmp2 = 400
    data_lognorm.append([tmp2, [tmp1]])

    tmp1 = np.random.uniform(3, 10, int(T)).cumsum()
    tmp2 = 150
    data_unif.append([tmp2, [tmp1]])

# Making tensorflow datasets

data_exp = create_dataset(data_exp)
data_lognorm = create_dataset(data_lognorm)
data_unif = create_dataset(data_unif)

TRAIN_SIZE = 0.8

data_exp_train = data_exp.take(int(TRAIN_SIZE * N))
data_exp_valid = data_exp.skip(int(TRAIN_SIZE * N))

data_lognorm_train = data_lognorm.take(int(TRAIN_SIZE * N))
data_lognorm_valid = data_lognorm.skip(int(TRAIN_SIZE * N))

data_unif_train = data_unif.take(int(TRAIN_SIZE * N))
data_unif_valid = data_unif.skip(int(TRAIN_SIZE * N))

# Setting up the model

def make_model():
    input = layers.Input(shape=(None, 2))
    x = LaggedSequence()(input)
    x = layers.LSTM(128, activation='tanh')(x)
    x = layers.Dense(64, activation='tanh')(x)
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(16, activation='tanh')(x)
    output = layers.Dense(1, activation='softplus')(x)
    model = IntensityRNN(inputs = input, outputs = output)
    return model

model_exp = make_model()
model_lognorm = make_model()
model_unif = make_model()

# Compiling the models with an optimizer

opt = keras.optimizers.Adam(lr=0.002, decay=0.0005)
model_exp.compile(optimizer=opt, L=50, d=1)
model_lognorm.compile(optimizer=opt, L=50, d=1)
model_unif.compile(optimizer=opt, L=50, d=1)

# Training the models

MAX_EPOCHS = 10

model_exp.fit(x=data_exp_train.batch(20), epochs=MAX_EPOCHS)
model_lognorm.fit(x=data_lognorm_train.batch(20), epochs=MAX_EPOCHS)
model_unif.fit(x=data_unif_train.batch(20), epochs=MAX_EPOCHS)

model_exp.save_weights('exp/weights', save_format='tf')
model_lognorm.save_weights('lognorm/weights', save_format='tf')
model_unif.save_weights('unif/weights', save_format='tf')
