import streamlit as st
import sys
sys.path.insert(1, '../')

from util.anomaly import AnomalyConfiguration, PointAnomaly, CollectiveAnomaly, SequentialAnomaly

def point_anom_gaussian_form(name):

    # Initialize session state for form fields and submission status
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
        st.session_state.percentage = None
        st.session_state.mu = None
        st.session_state.std = None
        st.session_state.num_values = None

    # Function to validate form fields
    def validate_form():
        if st.session_state.percentage is not None and st.session_state.mu is not None and st.session_state.std is not None:
            return True
        return False

    with st.form(key=f'{name}_form'):
        col1,col2 = st.columns(2)
        with col1:
            st.session_state.percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
            st.session_state.mu = st.number_input(label="Mean", key=f"{name}4", value=0.00)
        with col2:
            st.session_state.num_values = st.number_input(label='Number of Unique Values (Optional)',min_value=0, value=None, key=f"{name}1")
            st.session_state.std = st.number_input(label="Standard Deviation", key=f"{name}5", value=1.00)

        submit_button = st.form_submit_button(label='Create Anomaly Module')

    if submit_button:
        st.write(st.session_state.percentage, st.session_state.mu, st.session_state.std)
        if validate_form():
            st.success("Anomaly Module created successfully!")
            st.session_state.configured_module = PointAnomaly(percentage= st.session_state.percentage/100, mu= st.session_state.mu, std= st.session_state.std, num_values= st.session_state.num_values, distribution='gaussian')
            # Reset form submission state
            st.session_state.form_submitted = False
            return st.session_state.configured_module
        else:
            st.error("Please fill in all required fields to create this anomaly module.")
            st.session_state.form_submitted = True

if __name__ == "__main__":
    point_anom_gaussian_form('hi')