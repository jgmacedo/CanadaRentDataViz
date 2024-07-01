import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from data_viz_functions import create_price_chart_per_area, show_price_per_bedroom, show_price_per_parking, compute_and_plot_correlation_matrix

st.title('Canadian Rent Data Analysis :flag-ca:')
st.write('This web app allows you to analyze real estate data from Kijiji listings in Canada.'
         'The data was scraped using Selenium and bs4 in June 2024 and the listings were cleaned and saved as CSV files. '
         'Refer to my Github to see the source-code')

data_vancouver = pd.read_csv('data/cleaned/greater-vancouver-area-cleaned.csv')
data_toronto = pd.read_csv('data/cleaned/greater-toronto-area-cleaned.csv')
data_vancouver.drop('sqft', axis=1, inplace=True)
data_toronto.drop('sqft', axis=1, inplace=True)

# Charts
price_location_vancouver = create_price_chart_per_area(data_vancouver, 'Vancouver')
price_location_toronto = create_price_chart_per_area(data_toronto, 'Toronto')
price_parking_vancouver = show_price_per_parking(data_vancouver, 'Vancouver')
price_parking_toronto = show_price_per_parking(data_toronto, 'Toronto')
price_bedroom_vancouver = show_price_per_bedroom(data_vancouver, 'Vancouver')
price_bedroom_toronto = show_price_per_bedroom(data_toronto, 'Toronto')

st.write('Something to note: Outliers were not removed from the data, so the data may be skewed. Correlations were '
         'significantly affected by the missing outliers.')

# Visualizations of the data
vancouver_col, toronto_col = st.columns(2)
with vancouver_col:
    average_price = data_vancouver['price'].mean()
    st.metric('Average Price in Greater Vancouver Area', f"${average_price:,.2f}")
    st.metric('Number of Listings in Greater Vancouver Area', data_vancouver.shape[0])
    st.plotly_chart(px.histogram(data_vancouver, x='price', title='Price Distribution in Greater Vancouver Area', histnorm='density'))

with toronto_col:
    average_price = data_toronto['price'].mean()
    st.metric('Average Price in Greater Toronto Area', f"${average_price:,.2f}")
    st.metric('Number of Listings in Greater Toronto Area', data_toronto.shape[0])
    st.plotly_chart(px.histogram(data_toronto, x='price', title='Price Distribution in Greater Toronto Area', histnorm='density'))

st.write('Something to note: Outliers were not removed from the data, so the data may be skewed. Correlations were '
         'significantly affected by the missing outliers.')

vancouver_col2, toronto_col2 = st.columns(2)

with vancouver_col2:
    st.plotly_chart(price_location_vancouver, use_container_width=True)
    st.plotly_chart(price_parking_vancouver)
    st.plotly_chart(price_bedroom_vancouver, use_container_width=True)
    compute_and_plot_correlation_matrix(data_vancouver)

with toronto_col2:
    st.plotly_chart(price_location_toronto, use_container_width=True)
    st.plotly_chart(price_parking_toronto)
    st.plotly_chart(price_bedroom_toronto, use_container_width=True)
    compute_and_plot_correlation_matrix(data_toronto)
# Load the data for the heatmap

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from skopt import BayesSearchCV
from skopt.space import Integer
import matplotlib.pyplot as plt
from collections import OrderedDict
import plotly.graph_objects as go

# Load the data
df1 = pd.read_csv('data/cleaned/greater-toronto-area-cleaned.csv')
df2 = pd.read_csv('data/cleaned/greater-vancouver-area-cleaned.csv')
df = pd.concat([df1, df2], axis=0)

# Example feature selection (dropping 'price' column as target)
X = df.drop(columns=['price', 'link', 'title', 'description', 'Nearest intersection', 'location'])
y = df['price']

# Encoding categorical variables if any
X = pd.get_dummies(X)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Define the model
knn = KNeighborsRegressor()

# Define the parameter space for Bayesian Optimization
search_spaces = {
    'n_neighbors': Integer(1, 30),
    'weights': ['uniform', 'distance'],
    'p': Integer(1, 2)
}

# Implement Bayesian Optimization
bayes_search = BayesSearchCV(estimator=knn, search_spaces=search_spaces, n_iter=30, cv=3, random_state=42, n_jobs=-1)
bayes_search.fit(X_train, y_train)

# Best parameters from Bayesian Optimization
best_params_bayes = bayes_search.best_params_

param_grid = {
    'n_neighbors': [best_params_bayes['n_neighbors'] - 1, best_params_bayes['n_neighbors'], best_params_bayes['n_neighbors'] + 1],
    'weights': [best_params_bayes['weights']],
    'p': [best_params_bayes['p']]
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_params_grid = grid_search.best_params_

# Evaluate the model with the best parameters
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

results = OrderedDict({
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'R²': r2,
    'Adjusted R²': adj_r2
})

# K-Fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
knn_cv = KNeighborsRegressor(n_neighbors=best_params_bayes['n_neighbors'], weights=best_params_bayes['weights'], p=best_params_bayes['p'])

mse_scores = cross_val_score(knn_cv, X, y, scoring='neg_mean_squared_error', cv=kf)
mae_scores = cross_val_score(knn_cv, X, y, scoring='neg_mean_absolute_error', cv=kf)
r2_scores = cross_val_score(knn_cv, X, y, scoring='r2', cv=kf)

mse_cv = -np.mean(mse_scores)
rmse_cv = np.sqrt(mse_cv)
mae_cv = -np.mean(mae_scores)
r2_cv = np.mean(r2_scores)
adj_r2_cv = 1 - (1 - r2_cv) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

results['K-Nearest Neighbors'] = {
    'MSE': mse_cv,
    'RMSE': rmse_cv,
    'MAE': mae_cv,
    'R²': r2_cv,
    'Adjusted R²': adj_r2_cv
}

results_df = pd.DataFrame(results)

# Create a Streamlit app
st.title('Machine Learning Model Results')

st.write('## Best Parameters from Bayesian Optimization')
st.write(best_params_bayes)

st.write('## Best Parameters from Grid Search')
st.write(best_params_grid)

st.write('## Model Performance Metrics')
st.write(results_df)

# Add a histogram for the price distribution using Plotly
st.write('## Price Distribution')
fig_hist = px.histogram(df, x='price', nbins=30, title='Price Distribution')
st.plotly_chart(fig_hist)

# Add a scatter plot for predicted vs actual prices using Plotly
st.write('## Predicted vs Actual Prices')
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual'))
fig_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
fig_scatter.update_layout(title='Actual vs Predicted Prices', xaxis_title='Actual Prices', yaxis_title='Predicted Prices')
st.plotly_chart(fig_scatter)
