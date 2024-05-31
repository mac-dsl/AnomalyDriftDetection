import streamlit as st
import sys
sys.path.insert(1, '../')

from util.anomaly import AnomalyConfiguration, PointAnomaly, CollectiveAnomaly, SequentialAnomaly

def seq_anom_form(name):

    # Initialize session state for form fields and submission status
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
        st.session_state.percentage = None
        st.session_state.start = None
        st.session_state.end = None
        st.session_state.noise_factor = None
        st.session_state.length = None

    # Function to validate form fields
    def validate_form():
        if st.session_state.percentage is not None and st.session_state.noise_factor \
            is not None and st.session_state.length is not None:
            return True
        return False

    with st.form(key=f'{name}_form'):
        col1,col2 = st.columns(2)
        with col1:
            st.session_state.percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
            st.session_state.start = st.number_input(label='Starting Index (Optional)', min_value=0, value=None)
            st.session_state.length = st.number_input(label='Length of Anomaly', min_value=2, max_value=1000, key=f"{name}6", value=None)
        with col2:
            st.session_state.noise_factor = st.number_input(label="Noise Factor", value=None, min_value=0)
            st.session_state.end = st.number_input(label='Ending Index (Optional)', min_value=0, value=None)

        submit_button = st.form_submit_button(label='Create Anomaly Module')

    if submit_button:
        st.write(st.session_state.percentage, st.session_state.noise_factor, st.session_state.length)
        if validate_form():
            st.success("Anomaly Module created successfully!")
            st.session_state.configured_module = SequentialAnomaly(percentage=st.session_state.percentage/100, 
                                                                   length=st.session_state.length, start=st.session_state.start, 
                                                                   end=st.session_state.end, noise_factor=st.session_state.noise_factor)

            # Reset form submission state
            st.session_state.form_submitted = False
            return st.session_state.configured_module
        else:
            st.error("Please fill in all required fields to create this anomaly module.")
            st.session_state.form_submitted = True

if __name__ == "__main__":
    seq_anom_form('hi')