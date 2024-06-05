from scraperKijiji import scrape_listings, scrape_multiple_pages
import pandas

if __name__ == '__main__':
    url = 'https://www.kijiji.ca/b-apartments-condos/greater-vancouver-area/c37l80003?sort=dateDesc'
    gecko_path = 'geckodriver'
    listings = scrape_multiple_pages(url, gecko_path)

    for listing in listings:
        print(listing)
    listings = pandas.DataFrame(listings)
    listings.to_csv('listings.csv', index=False)