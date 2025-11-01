import time
import requests
from bs4 import BeautifulSoup, NavigableString
import re
import sqlite3

def extract_comment_with_quotes(content_div):

    output = []

    # Traverse all direct children
    for element in content_div.children:
        if element.name == 'blockquote':
            # Extract quote attribution if available
            citation = element.find('div', class_='ipsQuote_citation')
            citation_text = citation.get_text(strip=True) if citation else 'Quote:'
            
            # Extract the quoted contents
            quote_contents = element.find('div', class_='ipsQuote_contents')
            quote_text = '\n'.join(p.get_text(strip=True) for p in quote_contents.find_all('p')) if quote_contents else ''

            # Add formatted quote block
            output.append(f">>> {citation_text}\n{quote_text}")
        
        elif element.name == 'p':
            # Regular paragraph (non-quote)
            para = element.get_text(strip=True)
            if para:
                output.append(para)

    return '\n\n'.join(output)

def extract_response_data(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return []

    if response.status_code == 429:
        print(f"Received 429 Too Many Requests for {url}. Sleeping for 60 seconds...")
        time.sleep(60)
        return extract_response_data(url)  # Retry after sleep

    if response.ok:
        print("Successfully fetched the page!")
        # Extract data from the response
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return []

    parsed_data = {
        "post_id": None,
        "post_reply_id": None,
        "post_title": None,
        "post_time": None,
        "author": None,
        "post_link": url,
        "replies": []
    }
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    match = re.search(r'/topic/(\d+)', url)
    if match:
        parsed_data["post_id"] = match.group(1)
        print("Thread ID:", parsed_data["post_id"])
    else:
        print("No ID found.")

    page_number = int(soup.find('ul', class_='ipsPagination')['data-pages']) if soup.find('ul', class_='ipsPagination') else 1

    for i in range(page_number):
        if i == 0:
            temp = extract_first_page_data(url)
            parsed_data["post_reply_id"] = temp["post_reply_id"]
            parsed_data["post_title"] = temp["post_title"]        
            parsed_data["post_time"] = temp["post_time"]
            parsed_data["author"] = temp["author"]
            parsed_data["replies"].extend(temp["replies"])
        else:
            temp = extract_page_data(f"{url}page/{i+1}/")
            parsed_data["replies"].extend(temp)

    return parsed_data

    
def extract_first_page_data(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return []

    if response.status_code == 429:
        print(f"Received 429 Too Many Requests for {url}. Sleeping for 60 seconds...")
        time.sleep(60)
        return extract_first_page_data(url)  # Retry after sleep

    if not response.ok:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")
        return []
    
    parsed_data = {
        "post_reply_id": None,
        "post_title": None,
        "post_time": None,
        "author": None,
        "replies": []}

    soup = BeautifulSoup(response.text, 'html.parser')
    title_section = soup.find('div', class_='ipsPageHeader ipsResponsive_pull ipsBox ipsPadding sm:ipsPadding:half ipsMargin_bottom')
    parsed_data["post_title"] = title_section.find('h1', class_='ipsType_pageTitle ipsContained_container').text.strip()
    parsed_data["author"] = re.search(r'By\s*([^\s,]+)', title_section.find('strong').text.strip()).group(1) if title_section.find('strong') else "Unknown Author"
    parsed_data["post_time"] = title_section.find('time')['title'] if title_section.find('time') else "Unknown Time"

    parsed_data["replies"] = extract_page_data(url)
    parsed_data["post_reply_id"] = parsed_data["replies"][0]["reply_id"] if parsed_data["replies"] else None

    return parsed_data

    
def extract_page_data(url):
    parsed_data_set = []
    
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if not response.ok:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    all_replies = soup.find_all('article', class_= [
    "cPost", "ipsBox", "ipsResponsive_pull", "ipsComment",
    "ipsComment_parent", "ipsClearfix", "ipsClear",
    "ipsColumns", "ipsColumns_noSpacing", "ipsColumns_collapsePhone"
    ])

    print(f"Found {len(all_replies)} replies on page {url}")

    for reply in all_replies:
        raw_id = reply['id']  # elComment_2138760
        reply_id = raw_id.split('_')[-1]  # Extract the number

        side_info = reply.find('aside', class_= ["ipsComment_author", "cAuthorPane", "ipsColumn", "ipsColumn_medium", "ipsResponsive_hidePhone"])
        author = side_info.find('h3').text.strip() if side_info.find('h3') else "Unknown Author"

        main_section = reply.find('div', class_='ipsColumn ipsColumn_fluid ipsMargin:none')
        post_time = main_section.find('time')['title'] if main_section.find('time') else "Unknown Time"
        main_text = extract_comment_with_quotes(main_section.find('div', attrs={"data-role": "commentContent"}))


        parsed_data_set.append({
            "author": author,
            "reply_id": reply_id,
            "reply_time": post_time,
            "main_text": main_text})

    return parsed_data_set


def save_to_database(data):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('Data_Extraction/Database/Raw_Reviews.db')
    cursor = conn.cursor()
    
    # Create a table if it doesn't exist
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS escape_from_Tarkov_Official_Forum_Posts (
            post_id INTEGER PRIMARY KEY,
            post_reply_id INTEGER,
            post_title TEXT,
            post_time DATETIME,
            author TEXT,
            post_link TEXT
        )
    ''')    
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS escape_from_Tarkov_Official_Forum_Replies (
            author TEXT,
            reply_id INTEGER UNIQUE,
            reply_time DATETIME,
            main_text TEXT,
            post_id INTEGER,
            post_link TEXT
        )
    ''')
    
    # Insert data into the table
    cursor.execute(
        'INSERT OR REPLACE INTO escape_from_Tarkov_Official_Forum_Posts (post_id, post_reply_id, post_title, post_time, author, post_link) VALUES (?, ?, ?, ?, ?, ?)',
        (
            data['post_id'],
            data['post_reply_id'],
            data['post_title'],
            data['post_time'],
            data['author'],
            data['post_link']
        )
    )

    cursor.executemany(
        'INSERT OR REPLACE INTO escape_from_Tarkov_Official_Forum_Replies (author, reply_id, reply_time, main_text, post_id, post_link) VALUES (?, ?, ?, ?, ?, ?)',
        [
            (
                reply['author'],
                reply['reply_id'],
                reply['reply_time'],
                reply['main_text'],
                data['post_id'],
                data['post_link']
            ) for reply in data['replies']
        ]
    )

    # Commit changes and close the connection
    conn.commit()
    conn.close()

def normalize_thread_url(url):
    # Remove URL fragment (e.g., #comments)
    url = url.split('#')[0]
    # Remove trailing page/x
    url = re.sub(r'page/\d+/?$', '', url)
    return url

def get_thread_links(listing_url, pages):
    thread_links = set()  # Use set to avoid duplicates
    pattern = r"^https://forum\.escapefromtarkov\.com/topic/"

    for i in range(int(pages)):
        page_url = f"{listing_url}page/{i+1}/?sortby=start_date&sortdirection=desc"
        print(f"Fetching thread list from: {page_url}")
        response = requests.get(page_url, headers={"User-Agent": "Mozilla/5.0"})
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            threads = soup.find_all('h4', class_='ipsDataItem_title ipsContained_container')
            for thread in threads:
                a_tags = thread.findAll("a")
                for a_tag in a_tags:
                    if a_tag and a_tag.has_attr("href"):
                        raw_link = a_tag["href"]
                        normalized_link = normalize_thread_url(raw_link)
                        if re.match(pattern, normalized_link) and normalized_link not in thread_links:
                            print("Valid topic URL")
                            print("Thread link:", normalized_link)
                            thread_links.add(normalized_link)
        else:
            print(f"Failed to fetch thread listing page {i+1}")

    return list(thread_links)

WEBSITE_URL = input("input the URL of the website to scrape: ").strip()
pages = input("input the number of pages to scrape: ").strip()

thread_urls = get_thread_links(WEBSITE_URL, pages)

for url in thread_urls:
    print(f"Scraping {url}...")
    # try:
    response_text = extract_response_data(url)
    if isinstance(response_text, dict):
        save_to_database(response_text)
    else:
        print(f"Skipping {url} due to fetch error.")
    # except Exception as e:
    #     print(f"Error while processing {url}: {e}")


print("Data extraction and saving to database completed successfully.")

