import streamlit as st
from globals import ECG1

def main():
    st.title("Drift Injection")
    fig = ECG1.plot(start=0, end=2000)
    st.pyplot(fig)

if __name__ == "__main__":
    main()