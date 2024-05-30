import streamlit as st
import sys
sys.path.insert(1, '../')

from util.anomaly import AnomalyConfiguration, PointAnomaly, CollectiveAnomaly, SequentialAnomaly

def point_anom_uniform_form(name):

    # Initialize session state for form fields and submission status
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
        st.session_state.percentage = None
        st.session_state.lowerbound = None
        st.session_state.upperbound = None
        st.session_state.num_values = None

    # Function to validate form fields
    def validate_form():
        if st.session_state.percentage is not None and st.session_state.lowerbound is not None and st.session_state.upperbound is not None:
            return True
        return False

    with st.form(key='my_form'):
        col1,col2 = st.columns(2)
        with col1:
            st.session_state.percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
            st.session_state.lowerbound = st.number_input(label="Lowerbound Value", key=f"{name}3", value=None)
        with col2:
            st.session_state.num_values = st.number_input(label='Number of Unique Values (Optional)', min_value=0, value=None, key=f"{name}1")
            st.session_state.upperbound = st.number_input(label="Upperbound Value", key=f"{name}2", value=None)

        submit_button = st.form_submit_button(label='Create Anomaly Module')

    if submit_button:
        st.write(st.session_state.percentage, st.session_state.lowerbound, st.session_state.upperbound)
        if validate_form():
            st.success("Anomaly Module created successfully!")
            st.session_state.configured_module = PointAnomaly(percentage=st.session_state.percentage/100, 
                                                          lowerbound=st.session_state.lowerbound, num_values=st.session_state.num_values, upperbound=st.session_state.upperbound, distribution='uniform')
            # Reset form submission state
            st.session_state.form_submitted = False
            return st.session_state.configured_module
        else:
            st.error("Please fill in all required fields to create this anomaly module.")
            st.session_state.form_submitted = True

if __name__ == "__main__":
    point_anom_uniform_form('hi')