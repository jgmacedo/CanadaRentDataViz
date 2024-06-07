import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from data_viz_functions import create_price_chart_per_area, show_price_per_bedroom, show_price_per_parking, compute_and_plot_correlation_matrix

st.title('Canadian Rent Data Analysis :flag-ca:')
st.write('This web app allows you to analyze real estate data from Kijiji listings in Canada.''The data was scraped '
         'using Selenium and bs4 in June 2024 and the listings were cleaned and saved as CSV files. '
         'Refer to my Github to see the source-code')

data_vancouver = pd.read_csv('data/cleaned/greater-vancouver-area-cleaned.csv')
data_toronto = pd.read_csv('data/cleaned/greater-toronto-area-cleaned.csv')
data_vancouver.drop('sqft', axis=1, inplace=True)
data_toronto.drop('sqft', axis=1, inplace=True)

## charts

price_location_vancouver = create_price_chart_per_area(data_vancouver, 'Vancouver')
price_location_toronto = create_price_chart_per_area(data_toronto, 'Toronto')
price_parking_vancouver = show_price_per_parking(data_vancouver, 'Vancouver')
price_parking_toronto = show_price_per_parking(data_toronto, 'Toronto')
price_bedroom_vancouver = show_price_per_bedroom(data_vancouver, 'Vancouver')
price_bedroom_toronto = show_price_per_bedroom(data_toronto, 'Toronto')

st.write('Something to note: Outliers were not removed from the data, so the data may be skewed. Correlations were '
         'significantly affected by the missing outliers.')
## Visualizations of the data

vancouver_col, toronto_col = st.columns(2)
with vancouver_col:
    average_price = data_vancouver['price'].mean()
    st.metric('Average Price in Greater Vancouver Area', f"${average_price:,.2f}")
    st.metric('Number of Listings in Greater Vancouver Area', data_vancouver.shape[0])
    st.plotly_chart(px.histogram(data_vancouver, x='price', title='Price Distribution in Greater Vancouver Area'))

with toronto_col:
    average_price = data_toronto['price'].mean()
    st.metric('Average Price in Greater Vancouver Area', f"${average_price:,.2f}")
    st.metric('Number of Listings in Greater Toronto Area', data_toronto.shape[0])
    st.plotly_chart(px.histogram(data_toronto, x='price', title='Price Distribution in Greater Toronto Area'))

st.write('Something to note: Outliers were not removed from the data, so the data may be skewed. Correlations were '
         'significantly affected by the missing outliers.')

vancouver_col2, toronto_col2 = st.columns(2)


with vancouver_col2:
    st.plotly_chart(price_location_vancouver, use_container_width = True)
    st.plotly_chart(price_parking_vancouver)
    st.plotly_chart(price_bedroom_vancouver, use_container_width=True)
    compute_and_plot_correlation_matrix(data_vancouver)
with toronto_col2:
    st.plotly_chart(price_location_toronto, use_container_width=True)
    st.plotly_chart(price_parking_toronto)
    st.plotly_chart(price_bedroom_toronto, use_container_width=True)
    compute_and_plot_correlation_matrix(data_toronto)

