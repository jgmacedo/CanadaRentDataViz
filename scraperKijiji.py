from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def scrape_listings(driver):
    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Find all listing items
    listing_items = soup.select('[data-testid^="listing-card-list-item-"]')

    # Extract the required information
    listings_data = []
    for item in listing_items:
        price = item.select_one('[data-testid="listing-price"]').get_text(strip=True)
        link = item.select_one('[data-testid="listing-link"]')['href']
        title = item.select_one('[data-testid="listing-title"]').get_text(strip=True)
        location = item.select_one('[data-testid="listing-location"]').get_text(strip=True)
        description = item.select_one('[data-testid="listing-description"]').get_text(strip=True)

        attributes = {
            'Nearest intersection': None,
            'Bedrooms': None,
            'Bathrooms': None,
            'Unit type': None,
            'Parking included': None,
            'Size (sqft)': None
        }

        attribute_items = item.select('[data-testid^="re-attribute-list-"] li')
        for attribute_item in attribute_items:
            aria_label = attribute_item['aria-label']
            if aria_label in attributes:
                attributes[aria_label] = attribute_item.get_text(strip=True)

        listing_data = {
            'price': price,
            'link': link,
            'title': title,
            'location': location,
            'description': description,
            'attributes': attributes
        }

        listings_data.append(listing_data)

    return listings_data

def scrape_multiple_pages(url, gecko_path):
    # Initialize Selenium WebDriver
    service = Service(gecko_path)
    driver = webdriver.Firefox(service=service)

    # Open the initial URL
    driver.get(url)

    all_listings = []

    while True:
        # Wait until the listings are loaded
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid^="listing-card-list-item-"]'))
        )

        # Scrape the current page listings
        listings_data = scrape_listings(driver)
        all_listings.extend(listings_data)

        # Check for the "Next" button and click it
        try:
            # Wait until the "Next" button is clickable and click it
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'li[data-testid="pagination-next-link"] a'))
            )
            next_button.click()

            # Wait for the next page to load
            WebDriverWait(driver, 10).until(
                EC.staleness_of(next_button)
            )
        except Exception as e:
            print("No more pages or error:", e)
            break

    # Quit the driver
    driver.quit()

    return all_listings
