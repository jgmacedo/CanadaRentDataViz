from T.transforming_functions import remove_outliers
from T.transforming_functions import turn_price_to_float
from T.transforming_functions import turn_string_to_numbers
import pandas as pd
import ast


def clean_and_transform_data_kijiji(data):
    listings_df = pd.read_csv(data)
    listings_df['price'] = listings_df['price'].apply(turn_price_to_float)

    # Ensure 'attributes' column exists and contains stringified dictionaries
    if 'attributes' in listings_df.columns:
        listings_df['attributes'] = listings_df['attributes'].apply(ast.literal_eval)
        # Normalize the 'attributes' column to create separate columns for each attribute
        attributes_df = pd.json_normalize(listings_df['attributes'])
        # Concatenate the original dataframe with the new attributes dataframe
        listings_df = pd.concat([listings_df.drop(columns=['attributes']), attributes_df], axis=1)

    # Rename the column 'Size (sqft)' to 'sqft' if it exists
    if 'Size (sqft)' in listings_df.columns:
        listings_df.rename(columns={'Size (sqft)': 'sqft'}, inplace=True)

    # Remove any non-numeric characters and convert to numeric for 'sqft' if it exists
    if 'sqft' in listings_df.columns:
        listings_df['sqft'] = listings_df['sqft'].str.replace('sqft', '').str.replace(',', '').astype(float)
        # Create a new column 'sqm' by converting 'sqft' to square meters
        listings_df['sqm'] = listings_df['sqft'] * 0.092903

    if 'Bathrooms' in listings_df.columns:
        listings_df['Bathrooms'] = listings_df['Bathrooms'].apply(turn_string_to_numbers)

    if 'Bedrooms' in listings_df.columns:
        listings_df['Bedrooms'] = listings_df['Bedrooms'].apply(turn_string_to_numbers)

    if 'Parking included' in listings_df.columns:
        listings_df['Parking included'] = listings_df['Parking included']

    data_cleaned = listings_df.dropna()
    numeric_cols = ['price', 'Bedrooms', 'Bathrooms', 'sqft', 'sqm']

    # Remove outliers from the dataset
    data_no_outliers = remove_outliers(data_cleaned, numeric_cols)
    return data_no_outliers
