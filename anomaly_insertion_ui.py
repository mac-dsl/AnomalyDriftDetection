import streamlit as st
from util.anomaly import SequentialAnomaly, CollectiveAnomaly, PointAnomaly, AnomalyConfiguration
from util.anomaly_intervalsStream import *


def main():
    st.session_state.unmodified_data = st.session_state.ECG1
    st.title("Anomaly Injection")
    
    st.write(st.session_state.user_configured_anoms.anomalies)

    st.write(f"You have created {st.session_state.num_intervals} anomaly \
             intervals. Please specify which anomaly modules you would like inserted into each interval.")

    st.session_state.interval_config = []
    st.disable_inject_anom_btn = True
    
    with st.container(): 
        for i in range(int(st.session_state.num_intervals)):
            anom_interval = st.selectbox(label=f"Choose Anomaly Module for Interval {i+1}", options=st.session_state.user_configured_anoms.anomalies.keys() ,key=f"interval_anom_{i}", index=None)
            st.session_state.interval_config.append(anom_interval)
    
    st.write(st.session_state.interval_config)
    if not all(st.session_state.interval_config):
        st.warning("Please pick an anomaly module to inject for all intervals.")
        st.disable_inject_anom_btn = True
    else:
        st.disable_inject_anom_btn = False
        if(st.button("Inject Anomalies", disabled=st.disable_inject_anom_btn)):
            desired_anom_modules = [st.session_state.user_configured_anoms.anomalies[anom] for anom in st.session_state.interval_config]
            st.session_state.ECG_anomalies = createAnomalyIntervals(st.session_state.ECG1)
            st.session_state.ECG_anomalies.create_intervals(st.session_state.num_intervals, st.session_state.gap_size)
            st.session_state.ECG_anomalies.add_anomalies(*desired_anom_modules)
            col1, col2 = st.columns(2)
            with col1:
                start = st.number_input("Start", value=0, placeholder="Type a number...", min_value=0)
            with col2:
                end = st.number_input("End", value=len(st.session_state.ECG1.data), placeholder="Type a number...", max_value=len(st.session_state.ECG1.data), min_value=1)
            fig0 = st.session_state.unmodified_data.plot(start=start, end=end)
            fig1 = st.session_state.ECG1.plot(start=start, end=end)
            st.write("Unmodified Dataset")
            st.pyplot(fig0)
            st.write("Displaying injected Anomalies here")
            st.pyplot(fig1)


   
    # going to next page after anomaly injection (injecting drift)
    if st.button("Inject Drift"):
        st.session_state['current_page'] = 'Drift Injection'
        st.rerun()


if __name__ == "__main__":
    main()