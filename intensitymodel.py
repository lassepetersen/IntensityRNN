import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class IntensityRNN(keras.Model):
    """
    Class for nonparametric estimation of intensity functions
    using recurrent neural networks.

    The intensity function is modelled as

    $$
    \lambda(t | F_t) = \phi_\theta(t | F_t) 1(T < t)
    $$

    where $F_t$ is the history up to time $t$, $\phi(t | F_t)$ is the
    output of a RNN with positive output activation and $T$ is a
    censoring time.
    """


    def __init__(self, inputs = None, outputs = None, M_dim = None):
        """
        Layers used for the architechture
        """

        if inputs is not None:
            super(IntensityRNN, self).__init__(inputs=inputs, outputs=outputs)
            self.model_achitechture = "custom"
        else:
            super(IntensityRNN, self).__init__()
            self.model_achitechture = "default"
            self.M_dim = M_dim

            # Preprocessing layers
            self.preprocess_event_times = LaggedSequence()
            self.embed_event_type = EmbeddingWithNAN(
                input_dim=M_dim+1, output_dim=10
                )

            # Recurrent layers
            self.lstm1 = layers.LSTM(256, activation='tanh',
                                    return_sequences=True)
            self.lstm2 = layers.LSTM(128, activation='tanh')

            # Dense layers
            self.dense1 = layers.Dense(64, activation='tanh')
            self.dense2 = layers.Dense(32, activation='tanh')
            self.dense3 = layers.Dense(16, activation='tanh')
            self.dense4 = layers.Dense(1, activation='softplus')


    def call(self, inputs, training=True):
        """
        Forward pass method of the network. A sample has the form

            (X_1, z_1
            X_2, z_2,
            ..., 
            X_T, z_T,
            NAN, NAN,
            ..., 
            NAN, NAN, 
            t, z_0)

            with shape (time_steps, 2)

            The NAN-padding is used to give all
            sequences the same time length

            Here X_1, ..., X_T are the event times,
            z_1, ..., z_T are the event types,
            t is the point in time where we evaluate the intensity
            z_0:=0 is the event indicating trial start

        """

        if self.model_achitechture == "custom":
            return super(IntensityRNN, self).call(inputs=inputs)
        else:
            # Preprocessing event time
            x1 = self.preprocess_event_times(inputs)

            # Preprocessing event type (only taking events prior to t)
            x2 = self.embed_event_type(inputs)

            # Concatenating the results
            x = tf.concat(values=[x1, x2], axis=-1)

            # Preprocessing
            # x = self.preprocess(inputs)

            # Applying recurrent layers
            x = self.lstm1(x)
            x = self.lstm2(x)

            # Applying dense layers
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            outputs = self.dense4(x)

            return outputs


    def intensity(self, history, T):
        """
        Giving the estimated intensity function:

        $$
        \lambda(t | F_t) = \phi_\theta(t | F_t) 1(T < t)
        $$

        Arguments:
            data: array of the form
                (X_1, z_1
                 X_2, z_2,
                 ..., 
                 X_k, z_k)

            T: censoring times

        Returns
            intens: the intensity function as a function of time t
        """
        history = tf.constant(history, dtype=tf.float32)
        def intensity(t):
            t = tf.constant(t, dtype=tf.float32)
            tmp = tf.expand_dims(tf.concat([t, 0.0], axis=0), axis=0)
            input = tf.concat([history, tmp], axis=0)
            val = self(input)
            if t < T:
                return tf.reshape(val, shape=())
            else:
                return tf.constant(0.0, dtype=tf.float32)

        return intensity


    def compensator(self, history, T, approx=1000):
        """
        Giving the estimated compensator:

        $$
        Lambda(t | F_t) = \int_0^{min(t, T)} \lambda(s | F_s) ds
        $$

        Arguments:
            data: array of the form
                (X_1, z_1
                 X_2, z_2,
                 ..., 
                 X_k, z_k)
            T: censoring times
            approx: number of grid points for integral approximation

        Returns:
            compen: the compensator as a function of time t
        """

        intens = self.intensity(history, T)
        grid = tf.linspace(0.0, T, int(approx))[1:]
        grid_width = tf.constant(T / approx, dtype=tf.float32)
        eval = tf.map_fn(intens, grid)

        def compen(t):
            sub_eval = eval[grid <= t]
            return tf.reduce_sum(grid_width * sub_eval)

        return compen


    def compile(self, optimizer='rmsprop', loss=None, metrics=None,
                loss_weights=None, weighted_metrics=None,
                run_eagerly=None, L=50, a=1, **kwargs):

        """
        Compiling the model as usually, but specifying the parameter 
        L, which determines the approximation accuracy of the integral 
        part of the negative log-likelihood function, and a, which is 
        the index of the intensity process of interest.
        """

        super().compile(optimizer, loss, metrics, loss_weights,
                        weighted_metrics, run_eagerly, **kwargs)
        self.L = L
        self.a = a


    def train_step(self, data):
        """
        Performing a gradient step using the batch data

        Arguments:
            data: a batch of samples of the form

                (C, NAN
                X_1, z_1
                X_2, z_2,
                ..., 
                X_T, z_T,
                NAN, NAN,
                ..., 
                NAN, NAN)

                with shape (batch_size, time_steps, 2)

                The NAN-padding is used to give all
                sequences the same time length

                Here X_1, ..., X_T are the event times,
                z_1, ..., z_T are the event types,
                C is the censoring time

        Returns:
            loss: the batch loss
        """

        # Compute batch loss and track it with a gradient tape
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)

        # Compute gradient of loss wrt the trainable variables
        gradients = tape.gradient(loss, self.trainable_variables)

        # Perform a gradient step
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return the batch loss
        return {'train_loss': loss}


    def test_step(self, data):
        """
        Computing the loss of a validation data set

        Arguments:
             data: a batch of samples of the form

                (C, NAN
                X_1, z_1
                X_2, z_2,
                ..., 
                X_T, z_T,
                NAN, NAN,
                ..., 
                NAN, NAN)

                with shape (batch_size, time_steps, 2)

        Returns:
            loss: the batch loss
        """

        # Compute loss of validation batch
        val_loss = self.compute_loss(data)

        return {'loss': val_loss}


    def compute_loss(self, data):
        """
        Minus log-likelihood function for point processes:

        $$
        L(theta) = \int_0^{\infty} \phi_\theta(t \mid F_t) 1(t < C) dt
                   - \sum_{j: X_j < C, z_j = 1} \log \phi_\theta(X_j | F_{X_j})
        $$

        where \phi(t | F_t) is the output of a the RNN. The integral is
        approximated using a time discretization given by

        $$
        \int_0^{\infty} \phi_\theta(t \mid F_t) 1(t < C) dt
        \approx
        \sum_{l=1}^L Delta_l \cdot \phi_\theta(d_l | F_{d_l})
        $$

        where $0 = d_0 < d_1 < ... < d_L = C$ is an equidistant
        grid and $Delta_l = d_l - d_{l-1}$ is the grid_width.

        Arguments:
            data: batch tensor of censoring and event times
                  (C, X_1, ..., X_T, padding)

        Returns:
            loss: the minus log-likelihood of the current parameters
                given the batch
        """

        # Compute loss of each individual sample
        losses = tf.map_fn(fn = lambda D: self.comp1(D) - self.comp2(D), elems = data)

        # Compute the total loss over the batch
        total_loss = tf.reduce_sum(losses)

        return total_loss


    def comp1(self, D):
        """
        Perform forward passes over a grid from 0 to C of length L
        that is needed for computing the integral part of the loss function:

        $$
        \int_0^{\infty} \phi_\theta(t \mid F_t) 1(t < C) dt
        \approx
        \sum_{l=1}^L Delta_l \cdot \phi_\theta(d_l | F_{d_l})
        $$

        Arguments:
            D: tensor of censoring and event times (C, X_1, ..., X_T, padding)

        Returns:
            comp: tensor with the first component of the loss function
                for a single sample
        """

        # Extracting the censoring time
        C = D[0, 0]
        seq = D[1:, :]

        # Calculating the grid width
        grid_width = C / self.L

        # Creating grid to compute intensity over
        grid = tf.linspace(0., C, self.L+1)[1:]
        grid = tf.expand_dims(grid, axis=-1)

        # Compute forward passes over the input values
        eval = tf.map_fn(
            fn = lambda t: self(
                tf.concat([seq, tf.expand_dims(tf.concat([t, tf.zeros(1)], 0), 0)], 0)
                ), 
            elems = grid
        )

        # Computing the first element of the loss function for a single sample
        comp = tf.reduce_sum(grid_width * eval)

        return comp


    def comp2(self, D):
        """
        Perform forward passes over the event times strictly
        before the censoring time that are needed to compute the
        summation part of the loss function

        Arguments:
            D: tensor of censoring and event times (C, X_1, ..., X_T, padding)

        Returns:
            comp: tensor with the second component of the loss function
                for a single sample
        """
        # Extracting censoring and event sequence
        C = D[0, 0]
        seq = D[1:, :]

        # Indicator tensor which is 1 if
        # the j'th event is before censoring
        # and the event type is alpha,
        # and is 0 otherwise. Extremely important
        # for the @tf.function graph compilation
        N = seq.shape[0]
        tmp1 = seq[:, 0] < C
        tmp2 = seq[:, 1] == self.a
        tmp3 = tf.math.logical_and(tmp1, tmp2)
        Delta = tf.where(tmp3, tf.ones(N), tf.zeros(N))

        # Evaluating the network over the event times
        grid = tf.expand_dims(seq[:, 0], axis=-1)
        eval = tf.map_fn(
            fn = lambda t: self(
                tf.concat([seq, tf.expand_dims(tf.concat([t, tf.zeros(1)], 0), 0)], 0)
                ), 
            elems = grid
        )
        eval = tf.reshape(eval, shape=(-1,))

        # eval = tf.reshape(eval, shape=(-1))
        comp = tf.reduce_sum(Delta * tf.math.log(eval))

        return comp



def main():
    from datapreprocessing import create_dataset, sim_example_data
    from preprocessinglayers import LaggedSequence, EmbeddingWithNAN

    N = 200
    N_NODES = 10

    data = sim_example_data(N=N, n_nodes=N_NODES)
    dataset = create_dataset(data, marks = [1, 3, 4, 7])

    input = layers.Input(shape=(None, 2))
    x1 = LaggedSequence()(input)
    x2 = EmbeddingWithNAN(input_dim = 1 + N_NODES, output_dim = 10)(input)
    x = tf.concat([x1, x2], axis=-1)
    x = layers.LSTM(10)(x)
    x = layers.Dense(10, activation='tanh')(x)
    output = layers.Dense(1, activation='softplus')(x)
    model = IntensityRNN(inputs=input, outputs=output)
    model.compile(optimizer='adam', a = 3)
    model.fit(x=dataset.batch(25), epochs=5)
    


if __name__ == '__main__':
    main()

    # from datapreprocessing import create_dataset1, create_dataset, sim_example_data
    # from preprocessinglayers import LaggedSequence, EmbeddingWithNAN

    # input = layers.Input(shape=(None, 2))
    # x1 = LaggedSequence()(input)
    # x2 = EmbeddingWithNAN(input_dim=1+1, output_dim=3)(input)
    # x = tf.concat([x1, x2], axis=-1)
    # x = layers.LSTM(10)(x)
    # output = layers.Dense(1, activation='softplus')(x)
    # model = IntensityRNN(inputs=input, outputs=output)
    # model.compile(optimizer='adam', d = 1)
    
    # data = sim_example_data(N=50, n_nodes=4)
    # dataset = create_dataset1(data)

    # print(next(dataset.batch(1).as_numpy_iterator()))

    # opt = keras.optimizers.Adam(lr=0.002, decay=0.0005)
    # model.compile(optimizer=opt)

    # model.fit(x = dataset.batch(10), epochs=1)

    # scale = 1
    # N = 100
    # raw_data = []
    # for i in range(N):
    #     T = np.random.randint(low=3, high=10, size=1)
    #     seq = np.random.exponential(scale, T).cumsum()
    #     C = 0.9 * max(seq)
    #     raw_data.append([C, [seq]])

    # TRAIN_SIZE = 0.8
    # data_gen = create_dataset(raw_data)

    # print(next(data_gen.batch(1).as_numpy_iterator()))


    # data_gen_train = data_gen.take(int(TRAIN_SIZE * N))
    # data_gen_valid = data_gen.skip(int(TRAIN_SIZE * N))
    # opt = keras.optimizers.Adam(lr=0.002, decay=0.0005)
    # model.compile(optimizer=opt)
    #model.fit(x = data_gen_train.batch(10), epochs=2)


    if False:
        # Compatibility with 1-dimensional case, i.e., only one mark ------------------------
        
        # Data generation
        scale = 1
        N = 100
        raw_data = []
        for i in range(N):
            T = np.random.randint(low=3, high=10, size=1)
            seq = np.random.exponential(scale, T).cumsum()
            C = 0.9 * max(seq)
            raw_data.append([C, [seq]])

        TRAIN_SIZE = 0.8
        data_gen = create_dataset(raw_data)
        data_gen_train = data_gen.take(int(TRAIN_SIZE * N))
        data_gen_valid = data_gen.skip(int(TRAIN_SIZE * N))

        # Initializing the model
        model_1dim = IntensityRNN(M_dim=1, alpha=1)
        opt = keras.optimizers.Adam(lr=0.002, decay=0.0005)
        model_1dim.compile(optimizer=opt, L = 50, d = 1)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            min_delta=2,
            restore_best_weights=True)

        # Fitting the model to data
        model_1dim.fit(x = data_gen_train.batch(10), 
                    validation_data=data_gen_valid.batch(int((1-TRAIN_SIZE) * N)),
                    callbacks=[early_stop], 
                    epochs=2)

        # Fitting with more than one mark ---------------------------------------------------

        # Data
        scale = 1
        N = 100
        raw_data = []
        for i in range(N):
            T = np.random.randint(low=3, high=20, size=1)
            #seq1 = np.random.exponential(scale, T).cumsum()
            seq1 = np.random.uniform(low=0.5, high=0.15, size=T).cumsum()
            T = np.random.randint(low=2, high=5, size=1)
            seq2 = np.random.exponential(scale, T).cumsum()
            T = np.random.randint(low=2, high=5, size=1)
            seq3 = np.random.exponential(scale, T).cumsum()
            C = 0.9 * min(list(map(max, [seq1, seq2, seq3])))
            raw_data.append([C, [seq1, seq2, seq3]])

        TRAIN_SIZE = 0.8
        data_gen = create_dataset(raw_data)
        data_gen_train = data_gen.take(int(TRAIN_SIZE * N))
        data_gen_valid = data_gen.skip(int(TRAIN_SIZE * N))

        # Model initialization
        model_3dim = IntensityRNN(M_dim=3, alpha=1)
        opt = keras.optimizers.Adam(lr=0.002, decay=0.0005)
        model_3dim.compile(optimizer=opt)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            min_delta=2,
            restore_best_weights=True)
        
        model_3dim.fit(x = data_gen_train.batch(20), 
                validation_data=data_gen_valid.batch(int((1-TRAIN_SIZE) * N)),
                callbacks=[early_stop], 
                epochs=10)

        # Intensity plots -------------------------------------------------------------------
        import matplotlib.pyplot as plt

        dat = next(data_gen_valid.batch(1).as_numpy_iterator())[0]
        C = dat[0, 0]
        data = dat[1:, :]

        intens = model_3dim.intensity(data, C)

        compen = model_3dim.compensator(data, C)

        sq = np.linspace(0.0, C+1, 50+1)[1:]
        y = np.zeros(50)
        z = np.zeros(50)
        for i in range(50):
            y[i] = intens(sq[i])
            z[i] = compen(sq[i])

        plt.plot(sq, y)
        plt.savefig('intens.png')
        plt.plot(sq, z)
        plt.savefig('compen.png')