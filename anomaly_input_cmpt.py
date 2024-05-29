import streamlit as st
from util.anomaly import AnomalyConfiguration, PointAnomaly, CollectiveAnomaly, SequentialAnomaly


def anomaly_module_input_component(name, selected_anomaly, selected_dist_type,i):
    col1,col2 = st.columns(2)

    # to store the anomaly the user will configure
    st.session_state.configured_module = None

    if "Point Anomaly" == selected_anomaly and 'Uniform' == selected_dist_type:
        with col1:
            percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
            lowerbound = st.number_input(label="Lowerbound Value", key=f"{name,i}3", value=None)
        with col2:
            num_values = st.number_input(label='Number of Unique Values (Optional)', min_value=0, value=5, key=f"{name,i}1")
            upperbound = st.number_input(label="Upperbound Value", key=f"{name,i}2", value=None)
        
        if not (percentage and lowerbound and upperbound):
            st.warning("Please fill in all required parameters to create this anomaly module.")
        else:
            st.session_state.configured_module = PointAnomaly(percentage=percentage/100, 
                                                          lowerbound=lowerbound, num_values=num_values, upperbound=upperbound)
        

    if "Point Anomaly" == selected_anomaly and 'Gaussian' == selected_dist_type:
        with col1:
            percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
            mu = st.number_input(label="Mean", key=f"{name,i}4", value=1.00)
        with col2:
            num_values = st.number_input(label='Number of Unique Values (Optional)',min_value=0, value=5, key=f"{name,i}1")
            std = st.number_input(label="Standard Deviation", key=f"{name,i}5", value=1.00)
        
        if not (percentage and mu and std):
            st.warning("Please fill in all required parameters to create this anomaly module.")
        else:
            st.session_state.configured_module = PointAnomaly(percentage=percentage/100, mu=mu, std=std, num_values=num_values)

        
    if "Point Anomaly" == selected_anomaly and 'Skewed' == selected_dist_type: 
        col3, col4, col5 = st.columns(3)

        with col1: 
            percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
        with col2:
            num_values = st.number_input(label='Number of Unique Values (Optional)',min_value=0, value=None, key=f"{name,i}1")
        with col3:
            mu = st.number_input(label="Mean", key=f"{name,i}4", value=1.00)
        with col4:
            std = st.number_input(label="Standard Deviation", key=f"{name,i}5", value=1.00)
        with col5:
            skew = st.number_input(label="Skew Factor", key=f"{name,i}7", value=0.50)
        
        if not (percentage and mu and std and skew):
            st.warning("Please fill in all required parameters to create this anomaly module.")
        else:
            st.session_state.configured_module = PointAnomaly(percentage=percentage/100, num_values=num_values, mu=mu, std=std, skew=skew)
        

    if "Collective Anomaly" == selected_anomaly and 'Uniform' == selected_dist_type: 
        with col1:
            percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name)
            lowerbound = st.number_input(label="Lowerbound Value", key=f"{name,i}3", value=None)
            length = st.number_input(label='Length of Anomaly', min_value=2, max_value=1000, key=f"{name,i}9", value=None)
        with col2:
            num_values = st.number_input(label='Number of Unique Values', min_value=0, key=f"{name,i}1", value=5)
            upperbound = st.number_input(label="Upperbound Value", key=f"{name,i}2", value=None)

        if not (percentage and lowerbound and upperbound and length):
            st.warning("Please fill in all required parameters to create this anomaly module.")
        else:
            st.session_state.configured_module = CollectiveAnomaly(percentage=percentage/100, num_values=num_values, 
                                                               length=length,lowerbound=lowerbound, upperbound=upperbound)

    if "Collective Anomaly" == selected_anomaly and 'Gaussian' == selected_dist_type: 
        with col1:
            percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
            mu = st.number_input(label="Mean", key=f"{name,i}4", value=1.00)
            length = st.number_input(label='Length of Anomaly', min_value=0, max_value=1000,key=f"{name,i}9", value=None)
        with col2:
            print(f"{name,i}1")
            num_values = st.number_input(label='Number of Unique Values (Optional)',min_value=0, key=f"{name,i}1",value=5)
            std = st.number_input(label="Standard Deviation", key=f"{name,i}5", value=1.00)

        if not (percentage and mu and std and length):
            st.warning("Please fill in all required parameters to create this anomaly module.")
        else:
            st.session_state.configured_module = CollectiveAnomaly(percentage=percentage, mu=mu, 
                                                               length=length, num_values=num_values, std=std)

    if "Collective Anomaly" == selected_anomaly and 'Skewed' == selected_dist_type: 
        col3, col4, col5 = st.columns(3)

        with col1: 
            percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
            length = st.number_input(label='Length of Anomaly', min_value=0, max_value=1000, key=f"{name,i}6", value=None)
        with col2:
            num_values = st.number_input(label='Number of Unique Values (Optional)',min_value=0, key=f"{name,i}1", value=5)
        with col3:
            mu = st.number_input(label="Mean", key=f"{name,i}4", value=1.00)
        with col4:
            std = st.number_input(label="Standard Deviation", key=f"{name,i}5", value=1.00)
        with col5:
            skew = st.number_input(label="Skew Factor", key=f"{name,i}7", value=0.50)
        
        if not (percentage and mu and std and length and skew):
            st.warning("Please fill in all required parameters to create this anomaly module.")
        else:
            st.session_state.configured_module = CollectiveAnomaly(percentage=percentage/100, length=length, num_values=num_values, mu=mu, std=std, skew=skew)
    
    if "Sequential Anomaly" == selected_anomaly:
        with col1:
            percentage = st.number_input(label="Percentage of Anomalous Data (%)", min_value=0.0, max_value=100.0, key=name, value=None)
            start = st.number_input(label='Starting Index (Optional)', min_value=0, value=None)
            length = st.number_input(label='Length of Anomaly', min_value=2, max_value=1000, key=f"{name,i}6", value=None)
        with col2:
            noise_factor = st.number_input(label="Noise Factor", value=None, min_value=0)
            end = st.number_input(label='Ending Index (Optional)', min_value=0, value=None)
        
        if not (percentage and length and noise_factor):
            st.warning("Please fill in all required parameters to create this anomaly module.")
        else:
            st.session_state.configured_module = SequentialAnomaly(percentage=percentage/100, length=length, start=start, end=end, noise_factor=noise_factor)
        
    return st.session_state.configured_module


