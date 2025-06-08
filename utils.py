import pandas as pd
import numpy as np 

def pandas_loader(data_path, datetime_column = None):
    if datetime_column is None:
        return pd.read_csv(data_path)
    else:
        return pd.read_csv(data_path, parse_dates= [datetime_column])
    

def add_anomaly(signal, position = None, threshold = None, window = None):
    signal = np.array(signal).copy()
    if position is None:
        position = np.random.choice(len(signal))
    max_value = identify_mean(signal = signal, position = position, window = window)
    if threshold is None:
        threshold_list = np.arange(-0.5,0.5,0.001)
        threshold_list = threshold_list[threshold_list != 0]
        threshold = np.random.choice(threshold_list)
    signal[position] += threshold * max_value
    return signal



def identify_mean(signal, position, window):
    left = max(0, position - window)
    right = min(len(signal), position + window)
    return np.mean(signal[left:right])


