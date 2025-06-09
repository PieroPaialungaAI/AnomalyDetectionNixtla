import numpy as np 
from utils import * 
from plotter import * 
from constants import * 
import os
import pandas as pd
from dotenv import load_dotenv
from nixtla import NixtlaClient
import os

class AnomalyCalibrator:
    def __init__(self, processed_data, largest_threshold_size = 0.1, step_threshold = 0.001):
        self.input_data = processed_data
        self.input_signal = np.array(processed_data['y']).copy()
        self.n = len(self.input_signal)
        self.curr_threshold = largest_threshold_size
        self.step_threshold = step_threshold
        self.all_possible_locations = None

    def inject_anomaly(self, location=None, threshold=None, window=50):
        self.anomalous_signal = add_anomaly(
            signal=self.input_signal,
            location=location,
            threshold=threshold,
            window=window
        )
        return {'anomalous_signal': self.anomalous_signal, 'normal_signal': self.input_signal}


    def load_nixtla_client(self, nixtla_client = None):
        if nixtla_client is None:
            load_dotenv()  # looks for .env in current directory
            api_key = os.environ["NIXTLA_API_KEY"]
            nixtla_client = NixtlaClient(api_key=api_key)
        self.nixtla_client = nixtla_client


    def build_possible_locations(self,
                                 min_location = DEFAULT_MIN_LOCATION, max_location = DEFAULT_MAX_LOCATION, 
                                 num_signals = DEFAULT_NUM_LOCATION):
        if self.all_possible_locations is None:
            self.all_possible_locations = np.linspace(int(min_location*self.n), int(max_location*self.n), DEFAULT_DENSE_LOCATION)
        self.possible_locations = np.random.choice(self.all_possible_locations, size = num_signals).astype(int)
        

    def build_anomalous_dataset(self, min_location = DEFAULT_MIN_LOCATION, max_location = DEFAULT_MAX_LOCATION,
                                num_location = DEFAULT_NUM_LOCATION):
        if self.all_possible_locations is None:
            self.build_possible_locations(min_location, max_location, num_location)
        self.anomaly_dataset = np.zeros((len(self.possible_locations),self.n))
        i = 0
        for location in self.possible_locations:
            self.anomaly_dataset[i] = self.inject_anomaly(location = location, threshold = self.curr_threshold)['anomalous_signal']
            i += 1 


    def plot_anomalous_dataset(self):
        plt.figure(figsize = (10,5))
        for i in range(len(self.anomaly_dataset)):
            plt.plot(self.anomaly_dataset[i])
        plt.xlabel('Time (h)')
        plt.ylabel('Temperature (K)')
        plt.show()

