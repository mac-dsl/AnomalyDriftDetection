import streamlit as st
import sys
sys.path.insert(1, '../')

from util.anomaly import AnomalyConfiguration, PointAnomaly, CollectiveAnomaly, SequentialAnomaly