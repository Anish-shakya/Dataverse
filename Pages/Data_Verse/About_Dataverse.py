import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
def home_show():
    # First Section
    col1, col2 = st.columns(2)

    with col1:
        components.html("""
              <style>
        body {
            font-family: Arial, sans-serif;
            margin-left: 0;
            padding: 0;
            background-color: #ffffff;
            color: #333333;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
                        
        .section {
            padding: 40px 0;
        }

        .split-column {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
                        
        }

        .split-column .column {
            flex: 1;
            min-width: 45%;
            padding: 20px;
            margin-left:65px;
        }

        .heading {
            font-size: 3rem;
            font-weight: bold;
            color: #1a73e8;
        }

        .subheading {
            font-size: 2rem;
            margin-bottom: 10px;
            color: #1a73e8;
            text-align: center;
            font-weight:bold;
        }

        .description {
            font-size: 1.1rem;
            line-height: 1.6;
        }
        </style>         

        <div class="container">
                <!-- First Section -->
                <div class="section split-column">
                    <div class="column">
                        <div class="heading">DataVerse</div>
                        <div class="description">Your gateway to data-driven insights.</div>
                        <div class="description">Unlock the Power of Your Data Where Data Tells your Story</div>
                    </div>
                </div>
        </div> """,height=300)

    with col2:
        # Replace with your image path or URL
        st.image("Assets/img3.svg", width=500)
 
    with open('Pages/Data_Verse/Home.html','r',encoding='utf-8') as f:
        html_content = f.read()
    # HTML & CSS for the About Dataverse Section
    components.html(html_content,height=2100)
   