import numpy as np
from tick.hawkes import SimuHawkesExpKernels

# Simulating some data
n_nodes = 1
adjacency = 0.5 * np.ones((n_nodes, n_nodes))
decays = 3 * np.ones((n_nodes, n_nodes))
baseline = 0.25 * np.ones(n_nodes)
hawkes = SimuHawkesExpKernels(adjacency=adjacency,
                              decays=decays,
                              baseline=baseline,
                              verbose=False,
                              seed=123)
hawkes.end_time = 200
hawkes.track_intensity(0.01)

N = 500
raw_data = []
for i in range(N):
    hawkes.max_jumps = 15
    hawkes.simulate()
    events = hawkes.timestamps
    T = 0.8 * max(events[0])
    raw_data.append([T, events])
    hawkes.reset()

# RNN models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from intensitymodel import IntensityRNN, LaggedSequence
from datapreprocessing import create_dataset
import matplotlib.pyplot as plt
data = create_dataset(raw_data)

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

model = make_model()
opt = keras.optimizers.Adam(learning_rate=0.002, decay=0.0005)
model.compile(optimizer=opt, L=50, d=1)
model.fit(x=data.batch(20), epochs = 10)

# Vizualisation

test_hist = next(data.batch(10).as_numpy_iterator())[1]
T = test_hist[0, 0]
test_hist = test_hist[1:, :]

intensity = model.intensity(history = test_hist, T = T)

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
plt.title('Hawkes', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Estimated intensity', fontsize=16)
plt.tight_layout()
plt.savefig('plots/hawkes.pdf', format='pdf')
plt.close()

# Martingale residual plot
for i in range(10):
    test_hist = next(data.batch(10).as_numpy_iterator())[i]
    T = test_hist[0, 0]
    test_hist = test_hist[1:, :]
    events = test_hist[:, 0]
    compensator = model.compensator(history = test_hist, T = T, approx=200)
    def count_proc(t):
        return sum(events <= t)
    y = np.zeros(prec)
    sq = np.linspace(0.0, T+1, prec+1)[1:]
    for k in range(prec):
        y[k] = count_proc(sq[k]) - compensator(sq[k])
    plt.plot(sq, y)

plt.title('Hawkes', fontsize=20)
plt.xlabel('t', fontsize=16)
plt.ylabel('Martingale residual', fontsize=16)
plt.tight_layout()
plt.savefig('plots/MGR_hawkes.pdf', format='pdf')
plt.close()
