# CanGene: A Concept Drift and Anomaly Data Generator

## References of this repository
- https://moa.cms.waikato.ac.nz


## References
- Will be apeared

## Contributors
- Will be apeared

## Installation

Steps:

1. Clone the repository git (need to change the repo name to CanGene)

'''
git clone https://github.com/mac-dsl/AnomalyDriftDetection.git
'''

2. Install dependencies from requirement.txt

'''
pip install -r requirements.txt
'''

## Benchmark
All are datasets and time series are stored in ./data. We describe below the different types of datasets used in our benchmark.
1. ECG
- The ECG dataset is a standard electrocardiogram dataset which comes from the MIT-BIH Arrhythmia database, where anomalies represent ventricular premature contractions. (Downloaded from https://github.com/TheDatumOrg/TSB-UAD/tree/main)
  
2. Weather data
- The Weather dataset is a hourly, geographically aggregated temperature and radiation information in Europe originated from the NASA MERRA-2. The original Weather dataset contained timestamp, each country's temperature, and radiation information from 1960-to-2020, but for the demo, we separated each country's temperature data from 2017-to-2020 and saved them into '.arff' files in the repository. (From Open Power System Data, https://doi.org/10.25832/weather_data/2020-09-16:)

## Usage
We include several demo. file to test the experiments. 

1. Weather_Test.ipynb
- This is the exactly same test file for the draft which is introduced in 'References'. All the parameters should be described in 'demo_config.yaml' file, so users can augment and change it based on purpose.
- This test inject anomalies for 3-different weather data (GR, NO, DE) and generate drifts between them. 
- Details would be described both in the paper and jupyter notebook.
  
2. anomaly_injection_demo.ipynb
- 

3.  ECG_drift_demo.ipynb
- 

4.  moa_drift_generation.ipynb
- 
