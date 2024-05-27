import streamlit as st



def main():
    st.title("Anomaly Injection")
    
    col1, col2 = st.columns(2)

    with col1: 
        num_intervals = st.number_input(label="Number of Anomalous Intervals", min_value=1, value=None)
    with col2: 
        gap_size = st.number_input(label="Gap Size", min_value=1, value=None)
   
    fig = st.session_state.ECG1.plot()
    st.pyplot(fig)

    if num_intervals and gap_size:
        st.write("You can now create anomaly modules. How many would you like to create?")
        num_anomaly_modules = st.number_input(label="Number of Custom Anomaly Modules", min_value=1, max_value=10)


    


    # going to next page after anomaly injection (injecting drift)
    if st.button("Inject Drift"):
        st.session_state['current_page'] = 'Drift Injection'
        st.rerun()

if __name__ == "__main__":
    main()