import streamlit as st
from globals import ECG1

def main():
    st.title("Anomaly Injection")


    fig = ECG1.plot(start=0, end=2000)
    st.pyplot(fig)
    
    
    
    
    
    
    if st.button("Inject Drift"):
        st.session_state['current_page'] = 'Drift Injection'
        st.rerun()

if __name__ == "__main__":
    main()