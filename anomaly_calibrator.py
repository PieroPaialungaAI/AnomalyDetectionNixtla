import numpy as np 
from utils import * 
from plotter import * 

class AnomalyCalibrator:
    def __init__(self, signal):
        self.original_signal = np.array(signal).copy()

    def inject_anomaly(self, position=None, threshold=None, window=50):
        self.anomalous_signal = add_anomaly(
            signal=self.original_signal,
            position=position,
            threshold=threshold,
            window=window
        )
        return self.anomalous_signal

    def plot(self):
        plot_normal_and_anomalous_signal(self.original_signal, self.anomalous_signal)
