import streamlit as st

def main():
    st.title("Drift Injection")
    fig = st.session_state.ECG1.plot(start=0, end=2000)
    st.pyplot(fig)

if __name__ == "__main__":
    main()