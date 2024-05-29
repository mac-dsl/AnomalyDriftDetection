import streamlit as st
from util.anomaly import SequentialAnomaly, CollectiveAnomaly, PointAnomaly, AnomalyConfiguration

def main():
    st.title("Anomaly Injection")

    st.write(st.session_state.user_configured_anoms.anomalies)

    st.write(f"You have created {st.session_state.num_intervals} anomaly \
             intervals. Please specify which anomaly modules you would like inserted into each interval.")
    

    st.session_state.interval_config = []
    
    with st.container(): 
        for i in range(int(st.session_state.num_intervals)):
            anom_interval = st.selectbox(label=f"Choose Anomaly Module for Interval {i+1}", options=st.session_state.user_configured_anoms.anomalies.keys() ,key=f"interval_anom_{i}", index=None)
            st.session_state.interval_config.append(anom_interval)
    
    print(st.session_state.interval_config)
   
   
   
    # going to next page after anomaly injection (injecting drift)
    if st.button("Inject Drift"):
        st.session_state['current_page'] = 'Drift Injection'
        st.rerun()


if __name__ == "__main__":
    main()