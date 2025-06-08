import matplotlib.pyplot as plt 
import numpy as np 
from constants import * 

def plot_normal_and_anomalous_signal(signal, anomaly_signal, image_path = IMAGE_FOLDER):
    plt.figure(figsize = (10,6))
    plt.subplot(2,1,1)
    plt.plot(anomaly_signal, color ='darkorange', label = 'Anomalous Signal')
    plt.plot(signal, color ='navy', label = 'Non anomalous Signal')
    plt.xlabel('Time (t)')
    plt.ylabel('Temperature (y)')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.abs(anomaly_signal-signal), color ='k')
    plt.xlabel('Time (t)')
    plt.ylabel('|Signal - Anomalous Signal|')
    plt.tight_layout()
    plt.savefig(image_path + 'normal_vs_anomalous_signal.png')

