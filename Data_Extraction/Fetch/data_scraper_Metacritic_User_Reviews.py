from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import re
import sqlite3

def extract_response_data(response_text):
    parsed_data = []
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response_text, 'html.parser')

    all_reviews = soup.find_all('div', class_='c-siteReview g-bg-gray10 u-grid g-outer-spacing-bottom-large')

    reviews_data = []

    for review in all_reviews:
        review_text = review.find('div', class_='c-siteReview_quote g-outer-spacing-bottom-small').text.strip()
        if review_text == '[SPOILER ALERT: This review contains spoilers.]':
            continue
        score = review.find('span', attrs={'data-v-e408cafe': True}).text
        review_by = review.find('a', class_='c-siteReviewHeader_username g-text-bold g-color-gray90').text.strip()        
        review_time = review.find('div', class_='c-siteReviewHeader_reviewDate g-color-gray80 u-text-uppercase').text.strip()
        review_platform = review.find('div', class_='c-siteReview_platform g-text-bold g-color-gray80 g-text-xsmall u-text-right u-text-uppercase').text.strip()
        reviews_data.append([score,review_by,review_time,review_text, review_platform])
    return reviews_data

def save_to_database(data, game_name):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('Data_Extraction\Database\CS_Capstone.db')
    cursor = conn.cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {game_name}_metacritic_reviews (
            score REAL,
            author TEXT,
            time DATETIME,
            main_text TEXT,
            platform TEXT
        )
    ''')
    
    # Insert data into the table
    cursor.executemany(f'INSERT INTO {game_name}_metacritic_reviews (score, author, time, main_text, platform) VALUES (?, ?, ?, ?, ?)', data)
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()

game_name = input("input the game name: ").strip()
web_url = input("input the URL of the website to scrape: ").strip()

# Set up driver (headless optional)
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # Optional: for no GUI
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Open the page
driver.get(web_url)
time.sleep(5)  # wait for JavaScript to load

# Track how far we've scrolled
last_height = driver.execute_script("return document.body.scrollHeight")

# Scroll loop
scroll_attempts = 0
max_attempts = 100  # safety limit

while scroll_attempts < max_attempts:
    # Scroll near bottom
    driver.execute_script("window.scrollBy(0, 1500);")
    time.sleep(0.1)  # allow JS to fetch more content

    # Check if new content was added
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        scroll_attempts += 1  # no new content
    else:
        scroll_attempts = 0  # reset attempts
        last_height = new_height

# Save final HTML
html = driver.page_source
driver.quit()

print(html)  # For debugging, you can see the full HTML response

with open("metacritic_reviews.html", "w", encoding="utf-8") as f:
    f.write(html)

response_text = extract_response_data(html)
# fulllist.extend(response_text)

for item in response_text:
    print(f"SCORE: {item[0]}")
    print(f"AUTHOR: {item[1]}")
    print(f"TIME: {item[2]}")
    print(f"TEXT: {item[3]}")
    print(f"PLATFORM: {item[4]}")
    print("-" * 40)
# Save the extracted data to the database
save_to_database(response_text,game_name)

# print("Data extraction and saving to database completed successfully.")

