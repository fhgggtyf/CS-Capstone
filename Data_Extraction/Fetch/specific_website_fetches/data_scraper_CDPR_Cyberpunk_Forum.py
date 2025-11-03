import requests
from bs4 import BeautifulSoup, NavigableString
import re
import sqlite3

def extract_response_data(response_text):
    parsed_data = []
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response_text, 'html.parser')
    all_bb_wrappers = soup.find_all('div', class_='bbWrapper')
    all_headers = soup.find_all('div', class_='message-attribution-opposite')
    # Find all <a> tags with class 'reactionsBar-link'
    def has_all_classes(tag):
        return tag.name == 'article' and tag.has_attr('class') and all(cls in tag['class'] for cls in [
            'message', 'message--post', 'js-post', 'js-inlineModContainer'])

    articles = soup.find_all(has_all_classes)

    separated_divs = []

    post_number = []
    post_time = []

    reactions = []

    for article in articles:
        a_tag = article.find('a', class_='reactionsBar-link')\
        
        if a_tag is None:
            reactions.append(0)
            continue  # Skip if no reactions link found

        named_users = a_tag.find_all('bdi')
        num_named_users = len(named_users)  # 3

        # Extract number from "and 13 others" using regex
        text = a_tag.get_text()
        match = re.search(r'and (\d+) others', text)
        num_others = int(match.group(1)) if match else 0

        # Total reactions
        reactions.append(num_named_users + num_others)

    print(reactions)

    for header in all_headers:
        # Find the <a> and extract text (split on \n if needed)
        a_tag = header.find('a')
        post_number.append(a_tag.contents[0].strip())  # '#1'

        # Find the <time> and extract the 'title' attribute
        time_tag = soup.find('time')
        post_time.append(time_tag['title'] if time_tag and 'title' in time_tag.attrs else None)  # 'Dec 11, 2020 at 9:08 PM'

    for div in all_bb_wrappers:
        temp_soup = BeautifulSoup(str(div), 'html.parser')
        replied_to = temp_soup.find('div', class_='bbCodeBlock-expandContent js-expandContent')

        if replied_to is not None:
            separated_divs.append([post_number[0], post_time[0], replied_to, div, reactions[0]])
        else:
            separated_divs.append([post_number[0], post_time[0], None, div, reactions[0]])
        post_number.pop(0)
        post_time.pop(0)
        reactions.pop(0)

    for separated_div in separated_divs:
        quoted_text = separated_div[2] if separated_div[2] else None
        full_post_copy = separated_div[3]

        replied_text = quoted_text.get_text(separator='\n', strip=True) if quoted_text else None

        for quote in full_post_copy.find_all(['blockquote', 'div'], class_=["bbCodeBlock", "bbCodeBlock--quote", "bbCodeBlock-expandContent", "js-expandContent"]):
            quote.decompose()

        reply_text = full_post_copy.get_text(separator="\n", strip=True)
        separated_div[2]=replied_text
        separated_div[3]=reply_text

    # ...
    return separated_divs

def save_to_database(data, version_number):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('Data_Extraction/Database/Raw_Reviews.db')
    cursor = conn.cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS Cyberpunk_2077_Official_Forum_Reviews_{version_number} (
            post_num TEXT,
            post_time DATETIME,
            replied_text TEXT,
            main_text TEXT,
            upvote_number INTEGER
        )
    ''')
    
    # Insert data into the table
    cursor.executemany(f'INSERT INTO Cyberpunk_2077_Official_Forum_Reviews_{version_number} (post_num, post_time, replied_text, main_text, upvote_number) VALUES (?, ?, ?, ?, ?)', data)
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()

WEBSITE_URL = input("input the URL of the website to scrape: ").strip()
pages = input("input the number of pages to scrape: ").strip()
version_number = input("input the version number: ").strip()

fulllist = []

for i in range(int(pages)):
    print(f"Scraping page {i + 1}...")
    print(f"Fetching data from {WEBSITE_URL + f'page-{i + 1}'}...")

    response = requests.get(WEBSITE_URL + f'page-{i + 1}', headers={"User-Agent": "Mozilla/5.0"})

    if response.ok:
        print("Successfully fetched the page!")
        # save response text to a file
        with open(f'response{i}.html', 'w', encoding='utf-8') as file:
            file.write(response.text)
        print("Response saved to 'response.html'.")
        # Extract data from the response
        response_text = extract_response_data(response.text)

        fulllist.extend(response_text)

    else:
        print("Failed to fetch the page. Status code:", response.status_code)


for item in fulllist:
    print(f"Post Number: {item[0]}")
    print(f"Post Time: {item[1]}")
    print(f"Replied Text: {item[2]}")
    print(f"Reply Text: {item[3]}")
    print("-" * 40)
# Save the extracted data to the database
save_to_database(fulllist, version_number)

print("Data extraction and saving to database completed successfully.")

