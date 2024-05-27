import io
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from util.stream_streamlitcompatible import Stream
# importing the anomaly injection libraries
from util.anomaly_intervalsStream import *
from util.anomaly import CollectiveAnomaly, PointAnomaly, SequentialAnomaly
import numpy as np 
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from globals import ECG1

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("CANGene: Concept Drift and Anomaly Generator")

    uploaded_file = st.file_uploader("Choose an .arff or .csv file")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name
        
        print(file_path)
        ECG1 = Stream(file_path)
        start = st.number_input("Start", value=0, placeholder="Type a number...", min_value=0)
        end = st.number_input("End", value=len(ECG1.data), placeholder="Type a number...", max_value=len(ECG1.data), min_value=1)
        fig = ECG1.plot(start=start, end=end)
        st.pyplot(fig)

        # enable button to continue to anomaly injection here 
        if st.button("Inject Anomaly"):
            st.session_state['current_page'] = 'Anomaly Injection'
            st.rerun()


if __name__ == "__main__":
    main()