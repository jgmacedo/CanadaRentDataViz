import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st


def create_price_chart_per_area(data, city):
    """
    Creates a price distribution chart sorted by average price per area for the specified city.

    :param data: DataFrame containing the data with columns 'location' and 'price'.
    :param city: Name of the city for the chart title.
    """
    avg_price_per_area = data.groupby('location')['price'].mean().reset_index()

    avg_price_per_area = avg_price_per_area.sort_values(by='price', ascending=False)

    # Create the bar chart using the sorted average prices
    title = f'Average Price Distribution per Neighborhood in Greater {city} Area'
    price_chart_per_area = px.bar(avg_price_per_area, x='location', y='price', title=title)

    # Optionally, you can set labels and other properties
    price_chart_per_area.update_layout(
        xaxis_title='Neighborhood',
        yaxis_title='Average Price',
        xaxis={'categoryorder': 'total descending'}  # Ensure categories are sorted by the y-value
    )

    # Show the chart
    return price_chart_per_area


def show_price_per_bedroom(data, city):
    """
    Creates a box plot showing the price distribution per number of bedrooms for the specified city.

    :param data: DataFrame containing the data with columns 'Bedrooms' and 'price'.
    :param city: Name of the city for the chart title.
    """
    # Create the box plot
    title = f'Price Distribution per Number of Bedrooms in Greater {city} Area'
    price_per_bedroom = px.box(data, y='price', x='Bedrooms', title=title)

    # Show the chart
    return price_per_bedroom


def show_price_per_parking(data, city):
    """
    Creates a box plot showing the price distribution per number of parking spots for the specified city.

    :param data: DataFrame containing the data with columns 'Parking included' and 'price'.
    :param city: Name of the city for the chart title.
    """
    # Create the box plot
    title = f'Price Distribution per Number of Parking Spots in Greater {city} Area'
    price_per_parking = px.box(data, x='Parking included', y='price', title=title)

    # Show the chart
    return price_per_parking

def compute_and_plot_correlation_matrix(data):
    # Select numerical columns for the correlation matrix
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

    # Compute the correlation matrix
    correlation_matrix = data[numerical_cols].corr()

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    st.pyplot(plt)