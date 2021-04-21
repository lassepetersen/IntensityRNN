import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from intensitymodel import IntensityRNN, LaggedSequence, create_dataset
import matplotlib.pyplot as plt

# Loading the model weights
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

# Generating some test data
N = 20
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
data_exp_test = create_dataset(data_exp)
data_lognorm_test = create_dataset(data_lognorm)
data_unif_test = create_dataset(data_unif)

# Compiling and making a dummy training run to 
# re-initialize weights of the models

opt = keras.optimizers.Adam(lr=0.002, decay=0.0005)
model_exp.compile(optimizer=opt, L=50, d=1)
model_lognorm.compile(optimizer=opt, L=50, d=1)
model_unif.compile(optimizer=opt, L=50, d=1)

model_exp.fit(x = data_exp_test.take(1).batch(1))
model_lognorm.fit(x = data_lognorm_test.take(1).batch(1))
model_unif.fit(x = data_unif_test.take(1).batch(1))

# Loading back the weights of the saved models
model_exp.load_weights('exp/weights')
model_lognorm.load_weights('lognorm/weights')
model_unif.load_weights('unif/weights')

# Plotting

# Exps ------------------------------------------------------------------------
test_hist = next(data_exp_test.batch(10).as_numpy_iterator())[1]
T = test_hist[0, 0]
test_hist = test_hist[1:, :]

intensity = model_exp.intensity(history = test_hist, T = T)

prec = 1000
sq = np.linspace(0.0, T+1, prec+1)[1:]
y = np.zeros(prec)
for i in range(prec):
    y[i] = intensity(sq[i])

plt.plot(sq, y)
events = test_hist[:, 0]
for xx in events[events < T]:
    tmp = intensity(xx)
    plt.scatter(xx, tmp, color = 'red')
plt.title('Exponential', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Estimated intensity', fontsize=16)
plt.tight_layout()
plt.savefig('plots/exp.pdf', format='pdf')
plt.close()

# Martingale residual plot
for i in range(10):
    test_hist = next(data_exp_test.batch(10).as_numpy_iterator())[i]
    T = test_hist[0, 0]
    test_hist = test_hist[1:, :]
    events = test_hist[:, 0]
    compensator = model_exp.compensator(history = test_hist, T = T)
    def count_proc(t):
        return sum(events <= t)
    y = np.zeros(prec)
    sq = np.linspace(0.0, T+1, prec+1)[1:]
    for k in range(prec):
        y[k] = count_proc(sq[k]) - compensator(sq[k])
    plt.plot(sq, y)

plt.title('Exponential', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Martingale residual', fontsize=16)
plt.tight_layout()
plt.savefig('plots/MGR_exp.pdf', format='pdf')
plt.close()








# Lognorm ---------------------------------------------------------------------
test_hist = next(data_lognorm_test.batch(5).as_numpy_iterator())[1]
T = test_hist[0, 0]
test_hist = test_hist[1:, :]

intensity = model_lognorm.intensity(history = test_hist, T = T)

sq = np.linspace(0.0, T+1, prec+1)[1:]
y = np.zeros(prec)
for i in range(prec):
    y[i] = intensity(sq[i])

plt.plot(sq, y)
events = test_hist[:, 0]
for xx in events[events < T]:
    tmp = intensity(xx)
    plt.scatter(xx, tmp, color = 'red')
plt.title('Log-normal', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Estimated intensity', fontsize=16)
plt.tight_layout()
plt.savefig('plots/lognorm.pdf', format='pdf')
plt.close()

# Martingale residual plot
for i in range(10):
    test_hist = next(data_lognorm_test.batch(10).as_numpy_iterator())[i]
    T = test_hist[0, 0]
    test_hist = test_hist[1:, :]
    events = test_hist[:, 0]
    compensator = model_lognorm.compensator(history = test_hist, T = T)
    def count_proc(t):
        return sum(events <= t)
    y = np.zeros(prec)
    sq = np.linspace(0.0, T+1, prec+1)[1:]
    for k in range(prec):
        y[k] = count_proc(sq[k]) - compensator(sq[k])
    plt.plot(sq, y)

plt.title('Log-normal', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Martingale residual', fontsize=16)
plt.tight_layout()
plt.savefig('plots/MGR_lognorm.pdf', format='pdf')
plt.close()






# Unif ------------------------------------------------------------------------
test_hist = next(data_unif_test.batch(5).as_numpy_iterator())[1]
T = test_hist[0, 0]
test_hist = test_hist[1:, :]

intensity = model_unif.intensity(history = test_hist, T = T)

sq = np.linspace(0.0, T+1, prec+1)[1:]
y = np.zeros(prec)
for i in range(prec):
    y[i] = intensity(sq[i])

plt.plot(sq, y)
events = test_hist[:, 0]
for xx in events[events < T]:
    tmp = intensity(xx)
    plt.scatter(xx, tmp, color = 'red')
plt.title('Uniform', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Estimated intensity', fontsize=16)
plt.tight_layout()
plt.savefig('plots/unif.pdf', format='pdf')
plt.close()

# Martingale residual plot
for i in range(10):
    test_hist = next(data_unif_test.batch(10).as_numpy_iterator())[i]
    T = test_hist[0, 0]
    test_hist = test_hist[1:, :]
    events = test_hist[:, 0]
    compensator = model_unif.compensator(history = test_hist, T = T)
    def count_proc(t):
        return sum(events <= t)
    y = np.zeros(prec)
    sq = np.linspace(0.0, T+1, prec+1)[1:]
    for k in range(prec):
        y[k] = count_proc(sq[k]) - compensator(sq[k])
    plt.plot(sq, y)

plt.title('Uniform', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Martingale residual', fontsize=16)
plt.tight_layout()
plt.savefig('plots/MGR_unif.pdf', format='pdf')
plt.close()