import streamlit as st
from util.anomaly import AnomalyConfiguration, PointAnomaly, CollectiveAnomaly, SequentialAnomaly
from form_components.point_uniform import point_anom_uniform_form
from form_components.point_gaussian import point_anom_gaussian_form
from form_components.point_skewed import point_anom_skewed_form
from form_components.collective_uniform import collective_anom_uniform_form
from form_components.collective_gaussian import collective_anom_gaussian_form
from form_components.collective_skewed import collective_anom_skewed_form
from form_components.sequential import seq_anom_form


def anomaly_module_input_component(name, selected_anomaly, selected_dist_type,i):
   
    
    if "Point Anomaly" == selected_anomaly and 'Uniform' == selected_dist_type:
        st.session_state.configured_module = point_anom_uniform_form(name)
        if st.session_state.configured_module is not None:
            st.write(st.session_state.configured_module.__dict__)

    if "Point Anomaly" == selected_anomaly and 'Gaussian' == selected_dist_type:
        st.session_state.configured_module = point_anom_gaussian_form(name)
        if st.session_state.configured_module is not None:
            st.write(st.session_state.configured_module.__dict__)
        
    if "Point Anomaly" == selected_anomaly and 'Skewed' == selected_dist_type: 
        st.session_state.configured_module = point_anom_skewed_form(name)
        if st.session_state.configured_module is not None:
            st.write(st.session_state.configured_module.__dict__)
        
    if "Collective Anomaly" == selected_anomaly and 'Uniform' == selected_dist_type:
        st.session_state.configured_module = collective_anom_uniform_form(name)
        if st.session_state.configured_module is not None:
            st.write(st.session_state.configured_module.__dict__)

    if "Collective Anomaly" == selected_anomaly and 'Gaussian' == selected_dist_type:
        st.session_state.configured_module = collective_anom_gaussian_form(name)
        if st.session_state.configured_module is not None:
            st.write(st.session_state.configured_module.__dict__) 
        
    if "Collective Anomaly" == selected_anomaly and 'Skewed' == selected_dist_type: 
        st.session_state.configured_module = collective_anom_skewed_form(name)
        if st.session_state.configured_module is not None:
            st.write(st.session_state.configured_module.__dict__) 

    if "Sequential Anomaly" == selected_anomaly:
        st.session_state.configured_module = seq_anom_form(name)
        if st.session_state.configured_module is not None: 
            st.write(st.session_state.configured_module.__dict__)
    
    # if st.session_state.configured_module != None:
    #     return st.session_state.configured_module


