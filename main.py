from T.transform_data import clean_and_transform_data_kijiji

if __name__ == '__main__':
    url = (
        "https://www.kijiji.ca/b-apartments-condos/gta-greater-toronto-area/c37l1700272?sort=dateDesc")
    gecko_path = 'geckodriver'


    # filename = url.split('/')[-2] + '.csv'
    # listings = scrape_multiple_pages(url, gecko_path)
    # make_csv_file(listings, filename)
    # print("Process Ended")

    new_data = clean_and_transform_data_kijiji("data/gta-greater-toronto-area.csv")
    print(new_data.head())
    new_data.to_csv("data/cleaned/greater-toronto-area-cleaned.csv", index=False)
    print("Process Ended")





