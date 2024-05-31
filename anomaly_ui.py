import streamlit as st
from util.anomaly import SequentialAnomaly, CollectiveAnomaly, PointAnomaly, AnomalyConfiguration
from anomaly_input_cmpt import anomaly_module_input_component

 #storing the anomaly configuration
st.session_state.user_configured_anoms = AnomalyConfiguration()

def main():
    st.title("Anomaly Module Customization")
    
    col1, col2 = st.columns(2)

    with col1: 
        st.session_state.num_intervals = st.number_input(label="Number of Anomalous Intervals", min_value=1, value=None)
    with col2: 
        st.session_state.gap_size = st.number_input(label="Gap Size", min_value=1, value=None)
   
    #plot of the dataset
    fig = st.session_state.ECG1.plot()
    st.pyplot(fig)
    st.session_state.next_page = True

    #storing the anomaly configuration
    #st.session_state.user_configured_anoms = AnomalyConfiguration()


    if st.session_state.num_intervals and st.session_state.gap_size:
        st.write("You can now create anomaly modules. How many would you like to create?")
        num_anomaly_modules = st.number_input(label="Number of Custom Anomaly Modules", min_value=1, max_value=10)
        with st.container():
            for i in range(int(num_anomaly_modules)):
                st.session_state.disable_dist = True
                st.session_state.configured_module = None
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                st.session_state.disable_dist_option = False
                with col1: 
                    st.write(f'<p style="color:#164f91;">Custom Anomaly Module {i+1}', unsafe_allow_html=True)
                    name = st.text_input(label="Choose Module Name", key=f"anom_name{i}")
                    if name:
                        st.session_state.disable_dist = False
                with col2: 
                    pass
                with col3: 
                    selected_anomaly = st.selectbox(label="Select Anomaly Type", options=["Point Anomaly", "Collective Anomaly", "Sequential Anomaly"],index=None, key=f"type_anom{i}", disabled=st.session_state.disable_dist)
                    if selected_anomaly == 'Sequential Anomaly': 
                        st.session_state.disable_dist_option = True
                with col4: 
                    selected_dist_type = st.selectbox(label="Select Distribution Type", options=["Uniform", "Gaussian", "Skewed"], index=None, disabled=st.session_state.disable_dist_option or st.session_state.disable_dist, key=f"dist_anom{i}")
                
                # add custom component here
                if name and selected_anomaly:
                    if selected_dist_type or selected_anomaly == 'Sequential Anomaly':
                        anomaly_module_input_component(name, selected_anomaly,selected_dist_type,i)
                        if st.session_state.configured_module is not None:
                            st.write(st.session_state.configured_module.__dict__) 
                        else: 
                            st.write("The configured module is none.")
                        st.session_state.user_configured_anoms.add_anomaly_module(st.session_state.configured_module, name, selected_dist_type)
                        #st.write(st.session_state.user_configured_anoms.anomalies)
    
            if len(st.session_state.user_configured_anoms.anomalies) == num_anomaly_modules:
                # enabling next page button after everything is full
                st.session_state.next_page = False
    
    st.write(st.session_state.user_configured_anoms.anomalies)
    if st.button("Inject Anomaly Modules", disabled=st.session_state.next_page):
        # for debugging, remove later
        st.session_state['current_page'] = 'Anomaly Injection'
        st.rerun()
        
if __name__ == "__main__":
    main()