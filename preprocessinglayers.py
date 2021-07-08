import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Custom preprocessing layers

class RawSequence(keras.layers.Layer):
    """
    Custom layer giving

    (0, X_1, ..., X_M, t)

    where X_M is the largest event time smaller than t
    """

    def call(self, inputs):
        # Extracting the event times
        event_time = inputs[:, 0]
        
        # Removing the NAN-padding
        event_time = tf.boolean_mask(
            event_time, 
            tf.math.logical_not(tf.math.is_nan(event_time))
            )

        # Extracting the time point
        t = tf.expand_dims(event_time[-1], -1)

        # Looking at events prior to t
        prior_event = event_time[event_time < t]

        # Creating the raw sequence plus time point
        raw_event = tf.concat(values=[prior_event, t], axis=0)

        # Reshaping into something recurrent layers like
        output = tf.reshape(raw_event, shape=(1, -1, 1))

        return output


class LaggedSequence(keras.layers.Layer):
    """
    Custom layer giving

    (t - 0, t - X_1, ..., t - X_M)

    where X_M is the largest event time smaller than t
    """
    def call(self, inputs):
        # Extracting the event times
        event_time = inputs[:, 0]
        
        # Removing the NAN-padding
        event_time = tf.boolean_mask(
            event_time, 
            tf.math.logical_not(tf.math.is_nan(event_time))
            )

        # Extracting the time point
        t = event_time[-1]

        # Looking at events prior to t
        prior_event = event_time[event_time < t]

        # Adding an artificial 0 (trial start)
        prior_event = tf.concat(
            [tf.zeros(1), prior_event], axis=0
        )

        # Creating lags
        lagged_time = t - prior_event

        # Reshaping into something recurrent layers like
        output = tf.reshape(lagged_time, shape=(1, -1, 1))

        return output


class EmbeddingWithNAN(keras.layers.Embedding):
    def call(self, inputs):
        # Extracting event time and type
        event_time = inputs[:, 0]
        event_type = inputs[:, 1]

        # Removing the NAN-padding
        idx = tf.math.logical_not(tf.math.is_nan(event_time))
        event_time = tf.boolean_mask(event_time, idx)
        event_type = tf.boolean_mask(event_type, idx)

        # Extracting the event intensity time point
        t = event_time[-1]

        # Looking at prior event types
        prior_event_type = event_type[event_time < t]
        
        # Adding the zero event
        prior_event_type = tf.concat(
            [tf.zeros(1, dtype=tf.float32), prior_event_type], axis=0
        )
        
        embedded_event = super(EmbeddingWithNAN, self).call(
            prior_event_type
            )

        output = tf.expand_dims(embedded_event, axis=0)

        return output



def main():
    from datapreprocessing import sim_example_data, create_dataset1

    data = sim_example_data(N=10, n_nodes=10)
    dataset = create_dataset1(data, marks=[1, 3, 5, 6])

    tmp = next(dataset.batch(1).as_numpy_iterator())[0]
    print(tmp)
    print(EmbeddingWithNAN(input_dim=1+100, output_dim=2)(tmp))







if __name__ == "__main__":
    main()