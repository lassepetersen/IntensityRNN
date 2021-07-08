import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Predictable projection model
class PredProjRNN(keras.Model):
    """
    Class for nonparametric estimation of the 
    predictable projection process using 
    recurrent neural networks.
    """

    def compile(self, optimizer='rmsprop', loss=None, metrics=None,
                loss_weights=None, weighted_metrics=None,
                run_eagerly=None, R=50, b=1, phi_func=None, **kwargs):

        """
        Compiling the model as usual, but allowing for the additional
        argument L, which is used during training. d is the index of 
        the intensity process of interest. phi_func is the transformation 
        of the process N^b of interest.
        """

        super().compile(optimizer, loss, metrics, loss_weights,
                        weighted_metrics, run_eagerly, **kwargs)
        self.R = R
        self.b = b

        # If no phi function is provided, then we 
        # take the left limit of the counting process itself
        if phi_func is None:
            self.phi_func = self.__count_proc_left_limit__
        else:
            self.phi_func = phi_func



    def predictable_projection(self, history):
        """
        Method returning the estimated predictable 
        projection process.
        """
        def pred_proj(t):
            t = tf.constant(t, dtype=tf.float32)
            tmp = tf.expand_dims(tf.concat([t, 0.0], axis=0), axis=0)
            input = tf.concat([history, tmp], axis=0)
            return self.__call__(input)

        return pred_proj



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
        Computes batch loss

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
            loss: the loss of the batch
        """

        # Compute loss of each individual sample
        losses = tf.map_fn(fn = lambda D: self.loss_function(D), elems = data)

        # Compute the total loss over the batch
        total_loss = tf.reduce_mean(losses)

        return total_loss



    def loss_function(self, D):
        """
        The loss function of a single sample is given by

        $$
        L(theta) = \int_0^{T_j} (Z_{j, t} - \psi_{j, t})^2 dt
        $$

        where \psi_{j, t} is the output of an RNN. We approximate 
        the integral using a time discretization given by

        $$
        \int_0^{T_j} (Z_{j, t} - \psi_{j, t})^2 dt
        \approx 
        \sum_{r=1}^R \Delta_{j, r} \cdot 
        (Z_{j, \delta_{j, r}} - \psi_{j, \delta_{j, t}}))^2
        $$

        where $0 = d_{j,0} < d_{j,1} < ... < d_{j,R} = T_j$ is an equidistant
        grid and $Delta_{j,r} = d_{j,r} - d_{j,r-1}$ is the grid_width.

        Arguments:
            data: a sample of the form

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
            loss: the loss of the sample
        """

        # Extracting the censoring time
        C = D[0, 0]

        # Extracting the event history and 
        # separating into b and C
        hist = D[1:, :]
        idx_b = hist[:, 1] == self.b
        idx_C = hist[:, 1] != self.b
        
        hist_b = tf.boolean_mask(hist, idx_b)
        hist_C = tf.boolean_mask(hist, idx_C)
        
        # Building a grid from 0 to censoring time
        grid = tf.linspace(0., C, self.R+1)[1:]
        grid = tf.expand_dims(grid, axis=-1)

        # Calculating the grid width
        grid_width = C / self.R

        # Computing the values of the 
        # Z process over the grid values
        Z_values = tf.map_fn(fn = lambda t: self.phi_func(t, hist_b[:, 0]), 
                             elems = grid)

        # Computing forward passes over the grid values
        eval = tf.map_fn(
            fn = lambda t: self(
                tf.concat([hist_C, tf.expand_dims(tf.concat([t, tf.zeros(1)], 0), 0)], 0)
                ), 
            elems = grid
        )

        eval = tf.reshape(eval, shape=(-1,))
        loss = grid_width * tf.reduce_sum((Z_values - eval) ** 2)

        return loss



    def __count_proc_left_limit__(self, t, hist):
        counts = tf.reduce_sum(
                    tf.cast(hist < t, dtype=tf.float32)
                    )
        return counts



def main():
    from datapreprocessing import create_dataset, sim_example_data
    from preprocessinglayers import LaggedSequence, EmbeddingWithNAN
    from keras.callbacks import EarlyStopping

    N = 200
    N_NODES = 10

    # Simulating some data
    data = sim_example_data(N=N, n_nodes=N_NODES)
    dataset = create_dataset(data, marks = [1, 3, 4, 7])

    train_data = dataset.take(int(0.5 * N))
    valid_data = dataset.skip(int(0.5 * N)).take(int(0.25 * N))
    test_data = dataset.skip(int(0.5 * N)).skip(int(0.25 * N))

    # Fitting the model with early stopping according to validation loss
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        min_delta=0,
        restore_best_weights=True)

    # Network architecture
    input = layers.Input(shape=(None, 2))
    x1 = LaggedSequence()(input)
    x2 = EmbeddingWithNAN(input_dim=1+N_NODES, output_dim=10)(input)
    x = tf.concat([x1, x2], axis=-1)
    x = layers.LSTM(256, activation = 'tanh')(x)
    x = layers.Dense(128, activation = 'tanh')(x)
    x = layers.Dense(64, activation = 'tanh')(x)
    x = layers.Dense(32, activation = 'tanh')(x)
    x = layers.Dense(16, activation = 'tanh')(x)
    x = layers.Dense(8, activation = 'tanh')(x)
    output = layers.Dense(1, activation='linear')(x)
    model = PredProjRNN(inputs = input, outputs = output)

    opt = keras.optimizers.Adam(learning_rate=0.002, decay=0.0005)
    model.compile(R=50, optimizer=opt, b = 3)

    model.fit(x = train_data.batch(20), 
              validation_data = valid_data.batch(N), 
              epochs = 20)


if __name__ == "__main__":
    main()