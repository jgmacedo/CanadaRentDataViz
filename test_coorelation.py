from data_viz_functions import compute_and_plot_correlation_matrix
import pandas as pd

data_vancouver = pd.read_csv('data/cleaned/greater-vancouver-area-cleaned.csv')
data_toronto = pd.read_csv('data/cleaned/greater-toronto-area-cleaned.csv')

compute_and_plot_correlation_matrix(data_vancouver)



