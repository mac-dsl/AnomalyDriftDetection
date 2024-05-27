import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import streamlit as st

# injects a random sequence of anomalies (i.e collective anomaly)
class CollectiveAnomaly:
    def __init__(self, length: int, percentage: float, distribution: str = 'uniform', upperbound: float = None, lowerbound: float = None, mu: float = None, std: float = None, skew=None, num_values: int = 5) -> None:
        # if upperbound and lowerbound not given, then random sequence anomaly can be generated using random values already present in the data
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.length = length  # specifies the length of the random sequence
        self.percentage = percentage
        self.dist = distribution
        self.mean = mu
        self.std = std
        self.num_values = num_values
        self.skew = skew
        self.name = ''

# injects point anomalies; these deviate significantly from the rest of the data
class PointAnomaly:
    def __init__(self, percentage: float, distribution: str = 'uniform', mu: float = None, std: float = None, num_values: int = 5, lowerbound=None, upperbound=None, skew=None) -> None:
        self.percentage = percentage
        self.dist = distribution
        self.mean = mu
        self.std = std
        self.num_values = num_values
        self.skew = skew
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.name = ''

# injects sequential anomalies into the dataset (collective anomalies that keep repeating)
class SequentialAnomaly:
    def __init__(self, percentage: float, noise_factor: int, start=None, end=None, length=15):
        self.length = length
        self.percentage = percentage
        self.noise_factor = noise_factor
        self.start = start
        self.end = end
        self.name = ''

# used in the ui
class AnomalyConfiguration:
    def __init__(self, num_anomalies) -> None:
        self.num_anomalies = num_anomalies
        # key: name of anomaly, value: [type, distribution] 
        # type is either point, coll, sequential; distribution is either 'uniform' 'normal' 'skew'
        self.anomaly_dists = {} 
        # key: name of anomaly, value: the class object 
        self.anomalies = {}
        self.fields = {}
    
    def add_anomaly_module(self, anomaly, name:str, dist:str):
        if name in self.anomaly_dists:
            st.write("That name is already chosen. Please pick a different name.")
            return
        if type(anomaly) == PointAnomaly: 
            self.anomaly_dists[name] = ['Point Anomaly', dist]
        elif type(anomaly) == CollectiveAnomaly: 
            self.anomaly_dists[name] = ['Collective Anomaly', dist]
        elif type(anomaly) == SequentialAnomaly: 
            self.anomaly_dists[name] = ['Sequential Anomaly', dist]
        self.anomalies[name] = anomaly


    