# Anomaly Detection Performance Evaluation through Synthetic Anomalies

Let’s say you have a time series and an anomaly detection algorithm to detect anomalies. If you have a labeled dataset, the performance of your anomaly detection algorithm can be measured using metrics like precision, recall, or F1-score. But what if it’s not labeled?

## Synthetic Anomalies Detection
A way to assess the performance of your method is to manually inject **synthetic anomalies** (1) into your time series, as shown in the image below:

![Anomaly Example](notebook_images/anomaly_injection.png)

A successful outcome of the anomaly detection algorithm is shown in the following image: the algorithm correctly identifies the point where the anomaly was injected.

![Anomaly Example](notebook_images/anomaly_detection_output.png)

Of course, an unsuccessful outcome would be when the anomaly detector fails to identify the injected anomaly.

**These synthetic anomalies can be used to build a labeled dataset, as we now have full control over where the "true" anomalies occur.**
***
(1). Please note, an anomaly can configure in many different ways. Throughout this article, the anomaly is considered as a localized **spike** in the input time series. 
*** 

## Synthetic Anomalies Parameters

In the setup of synthetic anomalies, multiple **parameters** can influence anomaly detection:

1. The **kind** of injected anomalies: what do our injected anomalies look like?
2. The specific **anomaly detection algorithm** we are adopting.
3. The **size** of our anomalies: how "big" do we assume our anomalies to be?
4. The **location** of our anomalies in the time series.

### Fixed parameters

The first two parameters (**kind** and **anomaly detection algorithm**) are fixed in this blogpost.

#### Kind of Anomaly

As stated earlier, a reasonable assumption for the **"kind"** of anomaly is a localized **spike**. For example, in a weather dataset, where the amplitude (y-axis) represents temperature in Kelvin and the x-axis represents time in hours, the **spike** corresponds to a temperature that is significantly higher than average.

#### Anomaly Detection Algorithm

The **anomaly detection algorithm** that we will be testing is the ```TimeGPT-1``` model, developed by the [Nixtla](https://www.nixtla.io/) team. The idea is to use the **transformer** algorithm and conformal probabilities to get accurate predictions and uncertainty boundaries. You can read more about it in the original [paper](https://arxiv.org/abs/2310.03589), while another application of anomaly detection throught TimeGPT-1 can be found in this [blogpost](https://www.nixtla.io/blog/anomaly-detection-in-time-series-with-timegpt-and-python).


### Non Fixed Parameters/Variables
The **size** and **location** parameters are not fixed and will be considered as variables. An image showing injected anomalies at varying sizes and locations can be seen in the image below:


![table_first](notebook_images/table_first.png)

And the corresponding effect on the input time series is shown below:

![table_first](notebook_images/table_second.png)

## Performance Evaluation Method

### Goal of the method

Now, if you think about it, when the anomaly size is huge, a whatever anomaly detection model (even a very simple one) would be able to easily spot it. Nonetheless, when the anomaly size is almost zero, even a very powerful anomaly detection model would struggle to detect it.

The question we want to ask ourself is the following:
***"What is the smallest anomaly that we can detect through our algorithm?"***


### Evaluation Algorithm

The evaluation algorithm to detect the smallest detectable anomaly, which we are going to define as the **minimum detectable anomaly** is the following:
1. We fix the largest **size** of the anomaly (e.g. size = 0.1 $\times$ the average of the time series). 
2. We inject the anomaly, using the same size, at multiple locations in the time series.
3. For each time series with an injected anomaly, we run the anomaly detection algorithm and check whether the anomaly at the injected location is detected.
4. We measure performance across all locations using a metric such as: 
$$\text{Accuracy} = \frac{\text{Number of detected anomalies}}{\text{Number of anomalies}}$$
5. If the accuracy is satisfactory, we can reduce the size of the anomaly and repeat from step 2. If not, we interrupt the loop. 

A visual representation of this workflow can be seen in the following flowchart.

![Flowchart](notebook_images/workflow.png)

At the end of this loop, the smallest size that yields satisfactory accuracy will be defined as the **minimum detectable anomaly**.

## Performance Evaluation's Code

We are going to run the Performance Evaluation's algorithm described above on a real world dataset. The dataset and all the code that you need to run this example can be found in this [GitHub folder](https://github.com/PieroPaialungaAI/AnomalyDetectionNixtla), which you can clone using:

```bash
git clone https://github.com/PieroPaialungaAI/AnomalyDetectionNixtla.git
```



### Data

#### Data Source
The time series that we will be using come from a Kaggle open-source dataset, which can be found [here](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data). Note that you don't need to download anything, as the dataset is already available in the ```RawData``` folder. 

#### Raw Data

In this dataset, the x-axis is the hourly recorded time, and the y-axis is the temperature, measured in K. The loader for this dataset can be found in the  ```data.py``` script. The dataset provides multiple time series as columns (i.e., one column per city). For example, the last entries for the  time series for *Denver* look like this:

```python
from data import *
data = Data(datetime_column='datetime')
data.raw_data[['Denver', 'datetime']].tail()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Denver</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45248</th>
      <td>289.56</td>
      <td>2017-11-29 20:00:00</td>
    </tr>
    <tr>
      <th>45249</th>
      <td>290.70</td>
      <td>2017-11-29 21:00:00</td>
    </tr>
    <tr>
      <th>45250</th>
      <td>289.71</td>
      <td>2017-11-29 22:00:00</td>
    </tr>
    <tr>
      <th>45251</th>
      <td>289.17</td>
      <td>2017-11-29 23:00:00</td>
    </tr>
    <tr>
      <th>45252</th>
      <td>285.18</td>
      <td>2017-11-30 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>

#### Data Preprocessing 
Multiple preprocessing steps are now applied: 
1. We only deal with one time series, so we can pick the city. The default city is ```Phoenix``` (for no good reason, feel free to pick whatever city you like)
2. The time series is pretty long, so we isolate a portion of the time series to reduce time and power complexity. The default portion is between ```index = 41253``` and  ```index = 45253```. Again, feel free to change this however you'd like. Keep in mind that the full time series has length ```l = 45253``` (so don't exceed the boundaries)
3. In order to run our ```TimeGPT-1``` model on ```Nixtla```, we rename the datetime columns to ```ds``` and the timeseries amplitude to ```y```.


```python
data.isolate_city()
data.isolate_portion()
preprocessed_data = data.prepare_for_nixtla()
preprocessed_data.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>ds</th>
      <th>unique_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>297.96</td>
      <td>2017-06-16 09:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>297.15</td>
      <td>2017-06-16 10:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>295.80</td>
      <td>2017-06-16 11:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>295.00</td>
      <td>2017-06-16 12:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>294.33</td>
      <td>2017-06-16 13:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

This is the ```preprocessed_data``` that we will use for the rest of the blogpost. 

### Anomaly Calibrator
Everything related to anomaly injection and detection is implemented in the ```AnomalyCalibrator``` class. We will use the built-in functions from that object to explain the process. The input of ```AnomalyCalibrator``` is the ```preprocessed_data``` object from the ```Data``` class:

```python
from anomaly_calibrator import *
anomaly_calibrator = AnomalyCalibrator(processed_data = preprocessed_data)
```

#### Anomaly Injector
The function to inject an anomaly allows us to place the anomaly wherever we like. For example, we can inject an anomaly at location = 300 with size = 0.1 ($\times$ the time series average in a small window around location = 300) using the following line of code:

```python
anomaly_data = anomaly_calibrator.inject_anomaly(location = 300, threshold = 0.1)
plot_normal_and_anomalous_signal(anomaly_data['normal_signal'], anomaly_data['anomalous_signal'])
```


    
![png](temp_files/temp_22_0.png)
    

For a fixed anomaly size, we can generate ```num_location``` time series, each with the same anomaly ```size``` injected at a different location.


```python
anomaly_calibrator.build_anomalous_dataset(num_location = 20)
anomaly_calibrator.plot_anomalous_dataset()
```


    
![png](temp_files/temp_24_0.png)
    
#### Anomaly Detection Inference
Now we need to load Nixtla's API. To do that, make sure you have an API key. Follow the instructions[here](https://www.nixtla.io/docs/getting-started-setting_up_your_api_key). Once you have your API Key, store it in your system using:
```bash
export NIXTLA_API_KEY = "your_api_key"
```
Once you’ve followed the steps, you can easily import your Nixtla client using:


```python
anomaly_calibrator.load_nixtla_client()
```

So let's take this for a spin. Let's see how the Nixtla's anomaly detection algorithm performs for ```location = 800``` and ```size = 0.05```. 


```python
location = 800
threshold = 0.05
anomaly_data = anomaly_calibrator.inject_anomaly(location = location, threshold = threshold)
test_signal = anomaly_data['anomalous_signal']
anomaly_calibrator.run_anomaly_detection(test_signal)
ds_anomaly = preprocessed_data['ds'].loc[location]
anomaly_result = anomaly_calibrator.anomaly_result
anomaly_result[anomaly_result['ds'] == ds_anomaly]
```

    INFO:nixtla.nixtla_client:Validating inputs...
    INFO:nixtla.nixtla_client:Preprocessing dataframes...
    INFO:nixtla.nixtla_client:Calling Anomaly Detector Endpoint...





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>unique_id</th>
      <th>ds</th>
      <th>y</th>
      <th>TimeGPT</th>
      <th>TimeGPT-hi-99</th>
      <th>TimeGPT-lo-99</th>
      <th>anomaly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>664</th>
      <td>0</td>
      <td>2017-07-19 17:00:00</td>
      <td>320.59795</td>
      <td>304.36893</td>
      <td>309.00305</td>
      <td>299.7348</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

As we can see, the model correctly identifies the location where the anomaly was injected. Even visually, it's clear that the model successfully detects the anomaly.

```python
anomaly_calibrator.plot_anomaly_detection()
```


    
![png](temp_files/temp_30_0.png)
    
This is very promising, considering that the injected anomaly is only 5% of the average value of the time series.


#### Single Run

Now, given a fixed threshold, we can see the accuracy for a given location through the ```calibration_run``` function.


```python
anomaly_calibrator.calibration_run()
```

    Running TimeGPT for signal number 1, with location 1627, and size 0.1
    The anomaly has been detected
    Running TimeGPT for signal number 2, with location 3162, and size 0.1
    The anomaly has been detected
    Running TimeGPT for signal number 3, with location 1638, and size 0.1
    The anomaly has been detected
    Running TimeGPT for signal number 4, with location 3049, and size 0.1
    The anomaly has been detected
    Running TimeGPT for signal number 5, with location 416, and size 0.1
    The anomaly has been detected
    The accuracy for size 0.1 is 1.0





    1.0



Amazing! So for a size of an anomaly = 0.1, TimeGPT works extremely well. The full calibration loop is based on reducing the size of anomalies, so that we can see what is the minimum detectable size of anomaly. Let's work on it. 

### Anomaly Detection Calibration Loop
The following code makes you run the calibration loop described in the **"Anomaly Detection Performance Evaluation"** chapter. 
In order for the algorithm to run we would need a desired accuracy (the minimum accuracy we want to achieve). 


```python
calibration_dict = anomaly_calibrator.calibration_loop(number_of_runs = 20, desired_accuracy = 0.5)
```

    The accuracy for size 0.1 is 1.0
    The accuracy for size 0.09 is 1.0
    The accuracy for size 0.08 is 1.0
    The accuracy for size 0.07 is 1.0
    The accuracy for size 0.06 is 1.0
    The accuracy for size 0.05 is 1.0
    The accuracy for size 0.04 is 0.95
    The accuracy for size 0.03 is 1.0
    The accuracy for size 0.02 is 0.8
    The accuracy for size 0.01 is 0.1
    The accuracy for size 0.01 is 0.1, which is smaller than the desired accuracy 0.5
    The minimum detectable anomaly is 0.02


Now, as we can see, the minimum detectable anomaly is 0.02, but the accuracy there is still at 0.8. We can refine it by rerunning the calibration loop with a smaller starting size, as shown below:


```python
from anomaly_calibrator import AnomalyCalibrator
anomaly_calibrator = AnomalyCalibrator(processed_data = preprocessed_data, largest_threshold_size = 0.02, step_threshold = 0.001)
anomaly_calibrator.load_nixtla_client()
refinement_calibration_dict = anomaly_calibrator.calibration_loop(number_of_runs = 20, desired_accuracy = 0.5)
```

    The accuracy for size 0.02 is 0.7
    The accuracy for size 0.019 is 0.8
    The accuracy for size 0.018 is 0.7
    The accuracy for size 0.017 is 0.65
    The accuracy for size 0.016 is 0.45
    The accuracy for size 0.016 is 0.45, which is smaller than the desired accuracy 0.5
    The minimum detectable anomaly is 0.017


So if we are being more precise, the minimum detectable anomaly is 0.17. 
