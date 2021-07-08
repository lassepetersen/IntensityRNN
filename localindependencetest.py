import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import callbacks

from datapreprocessing import create_dataset, sim_example_data
from intensitymodel import IntensityRNN
from predprojmodel import PredProjRNN
from preprocessinglayers import LaggedSequence, EmbeddingWithNAN


class LocalIndependenceTest:
    """
    Nonparametric test for the hypothesis of conditional local independence.

    The test performs a test of whether N^a is conditionally 
    locally independent of N^b given N^C=(N^c)_{c \in C}, where 
    a, b and C are indices of a global counting process (N^d)_{d=1, ..., p}.
    """

    def __init__(self, data: pd.DataFrame, a: int, b: int, C: list) -> None:
        """
        Initializing the hypothesis test

        Parameters
        ----------
        data: pandas DataFrame
            tidy data frame with columns named "id", "event_time" and "event_mark"

        a: int
            the index of the counting process N^a
        
        b: int
            the index of the counting process N^b

        C: list
            the indices of counting processes N^C = (N^c)_{c \in C}. Must contain index a.
        """

        # Saving the data
        self.data = data
        self.N = data.shape[0]

        # Indices of the counting processes under interest
        self.a = a
        self.b = b
        self.C = C

        # Extracting the maximum mark
        self.max_mark = max(data["event_mark"])


    
    def train_intensity_model(self, optimizer: keras.optimizers.Optimizer=None, 
                              max_epochs: int=20, 
                              train_size: float=0.50,
                              valid_size: float=0.25, 
                              shuffle: bool=True) -> None:
        """
        Training an RNN model of the intensity function

        Parameters:
        ----------
        optimizer: keras.optimizers.Optimizer
            the optimizer used for training

        max_epochs: int
            the maximum number of epochs if training does 
            not stop due to early stopping

        train_size: float
            proportion of the data to be used for training

        valid_size: float
            proportion of the data to be used for validation

        shuffle: boolean
            whether the data should be shuffled between epochs
        """

        # Describing the architecture of the model
        input = layers.Input(shape=(None, 2))
        x1 = LaggedSequence()(input)

        # The embedding input dimension should be the maximum mark index and 
        # an additional dimension for the censoring. This might create redundant 
        # parameters, but these are not used anyway.
        x2 = EmbeddingWithNAN(input_dim=1+self.max_mark, output_dim=5)(input)

        x = tf.concat([x1, x2], axis = -1)
        x = layers.LSTM(128, activation='tanh')(x)
        x = layers.Dense(64, activation='tanh')(x)
        x = layers.Dense(32, activation='tanh')(x)
        x = layers.Dense(16, activation='tanh')(x)

        # Note that output activation must be non-negative
        output = layers.Dense(1, activation='softplus')(x)
        intensity_model = IntensityRNN(inputs = input, outputs = output)

        # Compiling the model with hyperparameters and optimizer
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=0.002, decay=0.0005)
            intensity_model.compile(optimizer=optimizer, a=self.a)
        else:
            intensity_model.compile(optimizer=optimizer, a=self.a)

        # Creating data object with the specified marks
        dataset = create_dataset(data=self.data, marks=[self.a] + self.C)

        # Shuffling the observations of the data set between epochs
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.N)

        # Data splitting
        self.train_data = dataset.take(int(train_size * self.N))
        self.valid_data = dataset.skip(int(train_size * self.N)).take(int(valid_size * self.N))
        self.test_data = dataset.skip(int(train_size * self.N)).skip(int(valid_size * self.N))

        # Fitting the model with early stopping according to validation loss
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            min_delta=0,
            restore_best_weights=True)

        intensity_model.fit(x=self.train_data.batch(25),
                            validation_data=self.valid_data.batch(int(valid_size * self.N)), 
                            callbacks=[early_stop], 
                            epochs=max_epochs)

        self.intensity_model = intensity_model



    def train_pred_proj_model(self, optimizer: keras.optimizers.Optimizer=None, 
                              max_epochs: int=20, 
                              train_size: float=0.50,
                              valid_size: float=0.25, 
                              shuffle: bool=True) -> None:
        """
        Training an RNN model of the predictable projection process

        Parameters:
        ----------
        optimizer: keras.optimizers.Optimizer
            the optimizer used for training

        max_epochs: int
            the maximum number of epochs if training does 
            not stop due to early stopping

        train_size: float
            proportion of the data to be used for training

        valid_size: float
            proportion of the data to be used for validation

        shuffle: boolean
            whether the data should be shuffled between epochs
        """

        # Describing the architecture of the model
        input = layers.Input(shape=(None, 2))
        x1 = LaggedSequence()(input)

        # The embedding input dimension should be the maximum mark index and 
        # an additional dimension for the censoring. This might create redundant 
        # parameters, but these are not used anyway.
        x2 = EmbeddingWithNAN(input_dim=1+self.max_mark, output_dim=5)(input)

        x = tf.concat([x1, x2], axis = -1)
        x = layers.LSTM(128, activation='tanh')(x)
        x = layers.Dense(64, activation='tanh')(x)
        x = layers.Dense(32, activation='tanh')(x)
        x = layers.Dense(16, activation='tanh')(x)

        # Note that output can be any real value
        output = layers.Dense(1, activation='linear')(x)
        pred_proj_model = PredProjRNN(inputs = input, outputs = output)

        # Compiling the model with hyperparameters and optimizer
        if optimizer is None:
            optimizer = keras.optimizers.Adam(learning_rate=0.002, decay=0.0005)
            pred_proj_model.compile(optimizer=optimizer, b=self.b)
        else:
            pred_proj_model.compile(optimizer=optimizer, b=self.b)

        # Creating data object with the specified marks
        dataset = create_dataset(data=self.data, marks=[self.b] + self.C)

        # Shuffling the observations of the data set between epochs
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.N)

        # Data splitting
        train_data = dataset.take(int(train_size * self.N))
        valid_data = dataset.skip(int(train_size * self.N)).take(int(valid_size * self.N))
        test_data = dataset.skip(int(train_size * self.N)).skip(int(valid_size * self.N))

        # Fitting the model with early stopping according to validation loss
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            min_delta=0,
            restore_best_weights=True)

        pred_proj_model.fit(x=train_data.batch(25),
                            validation_data=valid_data.batch(int(self.N)), 
                            callbacks=[early_stop], 
                            epochs=max_epochs)

        self.pred_proj_model = pred_proj_model



    def compute_test_stat_process():
        pass


    def __forward_pass_pred_proj__():
        pass









def main():
    N = 200
    N_NODES = 10
    data = sim_example_data(N=N, n_nodes=N_NODES)
    cli_test = LocalIndependenceTest(data=data, a=1, b=2, C=[1, 3, 5, 6])
    cli_test.train_intensity_model(max_epochs=1)
    cli_test.train_pred_proj_model(max_epochs=1)

    


if __name__ == "__main__":
    main()