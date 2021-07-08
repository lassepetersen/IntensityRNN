import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.backend import dtype
from tensorflow.python.ops.array_ops import repeat


def preprocess(raw):
    """
    Preprocessing of a single event sample.

    Arguments:
        raw: a list with the form (censoring time, event sequence type 1,
        ..., sequence type K). The censoring time is a float32 and the
        event sequences are 1 dim np array of increasing times.
        First sequence list element is the event of interest.

    Returns:
        dat: np array of shape (time_steps, 2) where first column is event
        time and second column is event type. First row is the censoring time.
    """

    # Pulling out censoring time and event list
    C = raw[0]
    X = raw[1:][0]

    # Helper array
    tmp = np.zeros((1, 2), dtype=np.float32)

    # Restructure event list data to array with event marks
    for i in range(len(X)):
        tmp = np.vstack(
            (tmp,
             np.vstack((X[i], np.repeat(i+1, len(X[i])))).transpose())
            )

    # Sorting according to event time
    idx = np.argsort(tmp[1:, :][:, 0])
    tmp = tmp[1:, ][idx, :]

    # Adding the censoring time as first element
    dat = np.vstack((np.array([[C, np.NAN]]), tmp)).astype('float32')

    return dat



def nan_padding(x, max_length):
    padding = np.repeat([np.NAN, np.NAN], max_length - x.shape[0])
    padding = padding.reshape(-1, 2)
    padding_seq = np.vstack((x, padding)).astype('float32')

    return padding_seq



def create_dataset_old(raw_data):
    tmp = list(map(preprocess, raw_data))
    max_length = max(list(map(np.shape, tmp)))[0]
    tmp = list(map(lambda x: nan_padding(x, max_length), tmp))
    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(tmp))

    return dataset



def create_dataset(data, marks):
    # Only taking the relevant marks (plus censoring)
    marks = marks + [0]
    idx = data["event_mark"].isin(marks)
    data = data[idx]
    del idx

    id, obs_length = np.unique(data["id"], return_counts=True)
    longest_seq = max(obs_length)

    tmp = []
    for i in id:
        # Extracting observation
        idx = data["id"] == i

        # Taking only event time and mark
        foo = data[["event_time", "event_mark"]]
        foo = foo[idx]

        # Separating censoring and events
        cens = foo[foo.event_mark == 0]
        cens.iloc[0, 1] = np.nan
        events = foo[foo.event_mark != 0]

        # Ordering the event times and adding the censoring
        events = events.sort_values("event_time")
        foo = cens.append(events)

        # Adding a NAN-padding to make all 
        # arrays the same length
        foo = nan_padding(foo, longest_seq)

        # Appending the preprocessed 
        # and NAN-padding data
        tmp.append(foo)

    # Creating the Dataset
    dataset = tf.data.Dataset.from_tensor_slices(tmp)

    return dataset



def sim_example_data(N, n_nodes):
    from tick.hawkes import SimuHawkesExpKernels

    adjacency = 0.5 * np.ones((n_nodes, n_nodes))
    adjacency[1, 0] = 0
    decays = 2 * np.ones((n_nodes, n_nodes))
    baseline = 0.1 * np.ones(n_nodes)
    hawkes = SimuHawkesExpKernels(adjacency=adjacency,
                                decays=decays,
                                baseline=baseline,
                                verbose=False,
                                seed=123)
    hawkes.end_time = 100
    hawkes.track_intensity(0.01)

    df = pd.DataFrame(columns=["id", "event_time", "event_mark"])
    for i in range(N):
        # Simulating from a the Hawkes process
        hawkes.max_jumps = np.random.randint(low=15, high=20)
        hawkes.simulate()

        # Extracting events times
        events = hawkes.timestamps
        event_time = np.concatenate(events)

        # Making subject ID
        id = np.repeat(i, len(event_time))

        # Making event types and censoring
        num_events = [len(x) for x in events]
        num_marks = len(num_events)
        mark_types = np.arange(1, num_marks+1)
        event_mark = np.repeat(mark_types, num_events)

        # Dataframe containing the current observation
        tmp1 = pd.DataFrame({"id": id, 
            "event_time": event_time, 
            "event_mark": event_mark}
            )

        # Making a censoring time
        cens_time = 0.8 * max(event_time)
        tmp2 = pd.DataFrame({"id": [i], 
            "event_time": [cens_time], 
            "event_mark": [0]}
            )

        # Concatenating to the global dataframe
        df = df.append([tmp1, tmp2], ignore_index=True)

        # Resetting the hawkes object to get a new seed
        hawkes.reset()
    
    return df



def main():
    x = sim_example_data(50, 7)
    data = create_dataset1(x, marks=[1])
    print(x)
    print(next(data.batch(10).as_numpy_iterator()))



if __name__ == "__main__":
    main()