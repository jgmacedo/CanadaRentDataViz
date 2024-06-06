import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.title('Canadian Real Estate Data Analysis :flag-ca:')
st.write('This web app allows you to analyze real estate data from Kijiji listings in Canada.')

data_vanc = pd.read_csv('data/greater-vancouver-area-cleaned.csv')
data_toronto = pd.read_csv('data/gta-greater-toronto-area-cleaned.csv')