import streamlit as st

class NepseAlphaChart:

    def showdetails():
        st.header("Nepse Alpha Chart 📊")


    def showchart():
        # URL of the website you want to load
        NepseAlphaChart.showdetails()
        url = "https://nepsealpha.com/trading/chart?symbol=NEPSE"

        # Display the website using an iframe
        st.markdown(f'<iframe src="{url}" width="100%" height="600"></iframe>', unsafe_allow_html=True)
