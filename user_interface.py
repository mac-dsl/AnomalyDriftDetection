import io
import streamlit as st
import pandas as pd
import numpy as np
import arff
import tempfile
import os
from util.stream_streamlitcompatible import Stream
# importing the anomaly injection libraries
from util.anomaly_intervalsStream import *
from util.anomaly import CollectiveAnomaly, PointAnomaly, SequentialAnomaly
import numpy as np 
import plotly.figure_factory as ff


def main():
    st.title("CANGene: Concept Drift and Anomaly Generator")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name
        
        print(file_path)
        ECG1 = Stream(file_path)
        ECG1.plot()
        

if __name__ == "__main__":
    main()