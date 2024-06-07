from T.transform_data import clean_and_transform_data_kijiji
from E.scraperKijiji import scrape_multiple_pages, make_csv_file

if __name__ == '__main__':
    url = ["https://www.kijiji.ca/b-apartments-condos/gta-greater-toronto-area/c37l1700272?sort=dateDesc",
           "https://www.kijiji.ca/b-apartments-condos/gta-greater-toronto-area/c37l1700272?sort=dateDesc"]
    gecko_path = 'geckodriver'

    # This code was written to scrape data from Kijiji and save it as a CSV file. The user can define multiple cities
    # to scrape data from.
    # With this project, I was able to learn a lot about ETL processes and how to scrape data from websites.
    # To show some data, I uploaded a web app to Streamlit so users can see the data in a more interactive way.


    filename = url.split('/')[-2] + '.csv'
    listings = scrape_multiple_pages(url, gecko_path)
    make_csv_file(listings, filename)
    print("Process Ended")

    new_data = clean_and_transform_data_kijiji("data/greater-vancouver-area.csv")

    new_data.to_csv("data/cleaned/greater-vancouver-area-cleaned.csv", index=False)
    print("Process Ended")





