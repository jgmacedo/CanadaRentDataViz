
# CanadaRentDataViz

## Description

**CanadaRentDataViz** is a data visualization project focused on analyzing rental information in Canada. The goal is to provide detailed insights into the real estate market, allowing users to visualize and interpret data effectively.

To see the project in the web, simply click this link https://canadarentdataviz.streamlit.app/

## Features

- **Data Scraping**: Collects rental listing information from a specific website.
- **Data Cleaning and Transformation**: Processes collected data to ensure consistency and usability.
- **Interactive Visualizations**: Creates charts and dashboards for data analysis.
- **Derived Calculations**: Generates calculated columns such as price per square meter, price per bedroom, and price per parking space.

## Technologies Used

- **Python**: Main language used for scraping, data cleaning, and creating visualizations.
- **Pandas**: Library for data manipulation and analysis.
- **Matplotlib**: Used for creating charts.
- **Beautiful Soup**: Library for data scraping.
- **Jupyter Notebook**: Interactive environment for development and data visualization.

## Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Contains Jupyter notebooks used for analysis and visualization.
- `scripts/`: Python scripts for data scraping and processing.
- `README.md`: This file, containing information about the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jgmacedo/CanadaRentDataViz.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CanadaRentDataViz
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows, use `venv\Scripts\activate`
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the scraping scripts to collect data:
   ```bash
   python scripts/scrape_data.py
   ```
2. Use Jupyter notebooks to clean, transform, and visualize the data. To start Jupyter Notebook, run:
   ```bash
   jupyter notebook
   ```
3. Open the desired notebook and follow the instructions to process and visualize the data.

## Contribution

If you wish to contribute to this project, feel free to open a pull request or report issues in the issues section.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
