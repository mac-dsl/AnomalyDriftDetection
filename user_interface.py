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

import home_ui, anomaly_ui, drift_ui, anomaly_insertion_ui

st.set_option('deprecation.showPyplotGlobalUse', False)

PAGES = {
    "Home": home_ui,
    "Anomaly Customization": anomaly_ui,
    "Anomaly Injection": anomaly_insertion_ui,
    "Drift Injection": drift_ui
}


if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'
    
def main():
        current_page = st.session_state['current_page']
        page = PAGES[current_page]
        page.main()

if __name__ == "__main__":
    main()

