"""
Metacritic User Review Scraper
==============================

This script fetches user reviews from a Metacritic game page across all available
platforms and writes the results into a SQLite database.  It uses Selenium to
drive a Chrome browser so that it can trigger Metacritic's infinite
scroll and load all reviews.  Reviews are filtered by a user‑supplied date
range (with an optional three‑year default) and only English reviews are
retained.  Duplicate reviews are ignored at insert time.

Usage
-----
Run the script and follow the prompts:

    $ python metacritic_scraper.py
    Enter the Metacritic game URL (e.g. https://www.metacritic.com/game/elden-ring/):
    Enter the start date (YYYY MM DD) or 'y' for default:
    Enter the end date (YYYY MM DD) or 'y' for default:

If both start and end dates are 'y', the script defaults to the last three
years of reviews (ending today).  Dates must be given in the format `YYYY MM
DD` separated by spaces.

Important:
  - Ensure you have Selenium and a compatible ChromeDriver installed and
    available on your `PATH` before running this script.  The script uses
    Selenium to drive a headless Chrome session.  See
    https://chromedriver.chromium.org/downloads for installation details.
  - The database file is created (if necessary) at
    `Data_Extraction/Database/CS_Capstone.db` relative to the current working
    directory.

The resulting table will be named in the form
`{game_slug}_{start}_{end}_metacritic` where `game_slug` comes from the URL and
`start`/`end` are dates formatted `YYYYMMDD`.
"""

import os
import re
import sqlite3
import string
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse

from bs4 import BeautifulSoup  # type: ignore

# Attempt to import Selenium; this will raise ImportError if Selenium is not
# installed.  The user should install Selenium and ChromeDriver locally.
try:
    from selenium import webdriver  # type: ignore
    from selenium.webdriver.chrome.options import Options  # type: ignore
except ImportError as exc:
    webdriver = None  # type: ignore
    Options = None  # type: ignore
    raise SystemExit(
        "Selenium is not installed. Install it via pip (pip install selenium) "
        "and ensure ChromeDriver is available on your PATH before running this script."
    )


def slugify_game(url: str) -> str:
    """Extract and sanitize the game slug from a Metacritic game URL.

    Example:
        https://www.metacritic.com/game/elden-ring/ -> elden_ring

    Parameters
    ----------
    url : str
        The full Metacritic game URL.

    Returns
    -------
    str
        A lowercase slug with non‑alphanumeric characters replaced by
        underscores.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')  # remove trailing slash
    # The slug is the final part of the path after `/game/`
    slug = path.split('/game/')[-1]
    # Replace hyphens and other invalid characters with underscore
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
    return slug.strip("_").lower()


def sanitize_table_name(name: str) -> str:
    """Sanitize a SQLite table name by replacing invalid characters.

    SQLite table names cannot contain spaces or special characters.  This
    function replaces any non‑alphanumeric character with an underscore and
    collapses consecutive underscores.

    Parameters
    ----------
    name : str
        Raw table name.

    Returns
    -------
    str
        Sanitized table name.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def parse_date(date_str: str) -> datetime.date:
    """Parse a review date string into a `datetime.date`.

    Metacritic user review dates are formatted like "Aug 14, 2025".  This
    function converts such strings into a `datetime.date`.  If parsing fails,
    a `ValueError` will be raised.

    Parameters
    ----------
    date_str : str
        Date string from the page (e.g. "Aug 14, 2025").

    Returns
    -------
    datetime.date
        The parsed date.
    """
    return datetime.strptime(date_str.strip(), "%b %d, %Y").date()


def is_english(text: str, threshold: float = 0.9) -> bool:
    """Heuristically determine whether a review text appears to be English.

    This function counts the proportion of ASCII characters (letters,
    digits, punctuation and whitespace) within the text.  If the
    proportion of ASCII characters is greater than or equal to `threshold`,
    the text is assumed to be English.  This is a very rough heuristic and
    may misclassify some reviews; however, external language detection
    libraries are not available in this environment.

    Parameters
    ----------
    text : str
        The review text.
    threshold : float
        Minimum fraction of ASCII characters required to consider the text
        English.  Defaults to 0.9 (90%).

    Returns
    -------
    bool
        True if the text appears to be English, False otherwise.
    """
    if not text:
        return False
    ascii_chars = 0
    for ch in text:
        # Consider ASCII letters, digits, punctuation and whitespace as English
        if ch in string.ascii_letters or ch in string.digits or ch in string.punctuation or ch.isspace():
            ascii_chars += 1
    ratio = ascii_chars / len(text)
    return ratio >= threshold


def extract_platform_codes(page_source: str) -> list[str]:
    """Extract unique platform codes from a Metacritic page source.

    Platform codes are found in URL parameters such as `platform=pc` or
    `platform=ps5`.  This function uses a regular expression to find all
    occurrences and returns a list of unique codes.

    Parameters
    ----------
    page_source : str
        HTML source of the page as returned by Selenium's `driver.page_source`.

    Returns
    -------
    list[str]
        A list of unique platform codes (e.g. ['pc', 'ps5', 'xbox-series-x']).
    """
    # Look for `platform=...` occurrences.  Capture letters, numbers and hyphens.
    codes = re.findall(r"platform=([a-zA-Z0-9\-]+)", page_source)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_codes: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)
    return unique_codes


def _find_review_blocks(soup: BeautifulSoup) -> list:
    """Locate review containers in the parsed HTML.

    Newer versions of Metacritic mark each review with a data-testid of
    ``product-review``.  When this attribute is present we can
    confidently select those elements.  If no such elements exist (e.g.,
    for older saved pages), we fall back to heuristics based on class
    names and simple text patterns to locate reviews.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML of the page.

    Returns
    -------
    list
        A list of elements that likely represent individual reviews.
    """
    """Locate review containers in the parsed HTML.

    In the current Metacritic markup, each user review is wrapped in a
    ``<div>`` with a ``data-testid="product-review"`` attribute.  When
    present, those elements uniquely identify review blocks and should
    always be used.  For compatibility with older pages or saved
    HTML files that may not include the ``data-testid`` attribute, we
    provide fallbacks.  First, we look for the legacy ``c-siteReview``
    container class.  As a last resort we employ a heuristic: any
    element whose class name contains ``review`` and whose text
    includes both a date (e.g. ``Aug 15, 2025``) and a numeric score.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML of the page.

    Returns
    -------
    list
        A list of elements that likely represent individual reviews.
    """
    # Prefer explicit data-testid markers if available.  These are
    # consistently used on contemporary Metacritic pages and saved files.
    blocks = soup.find_all('div', attrs={'data-testid': 'product-review'})
    if blocks:
        return blocks
    # Fall back to legacy selector used by older Metacritic markup
    blocks = soup.find_all('div', class_='c-siteReview')
    if blocks:
        return blocks
    # Generic heuristic: any element whose class contains 'review' and
    # includes a date and a score.
    def looks_like_review(tag: object) -> bool:
        if not hasattr(tag, 'name'):
            return False
        if tag.name not in {'div', 'article', 'li', 'section'}:
            return False
        classes = tag.get('class') or []
        if any('review' in c.lower() for c in classes):
            text = tag.get_text(" ", strip=True)
            if re.search(r"[A-Za-z]{3} \d{1,2}, \d{4}", text) and re.search(r"\b(10|[0-9])\b", text):
                return True
        return False
    return soup.find_all(looks_like_review)


def scroll_until_date(driver: webdriver.Chrome, start_date: datetime.date) -> None:
    """
    Scroll DOWN in fixed 1000px steps from the top and stop ONLY when:
      (A) the last review currently in the DOM is older than `start_date`, OR
      (B) the browser viewport has reached the very bottom of the page.

    We ONLY check the last review's date **when the page's total height changes**,
    i.e., after new content is appended and `document.body.scrollHeight` increases.
    """
    import re
    from bs4 import BeautifulSoup

    scroll_step = 1000
    pause_s = 1.5

    # Start at the very top.
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.8)

    last_height = driver.execute_script("return document.body.scrollHeight")
    current_y   = 0

    print(f"[DEBUG] start: docHeight={last_height}, start_date={start_date}")

    while True:
        # Compute bottom-of-page condition using viewport height + current Y offset.
        inner_h   = driver.execute_script("return window.innerHeight")
        page_y    = driver.execute_script("return window.pageYOffset")
        doc_h     = driver.execute_script("return document.body.scrollHeight")
        at_bottom = (page_y + inner_h) >= doc_h - 2  # tiny tolerance

        if at_bottom:
            print(f"[DEBUG] reached bottom: pageYOffset({page_y}) + innerHeight({inner_h}) >= docHeight({doc_h})")
            break  # condition (B)

        # Scroll by a fixed 1000px chunk.
        current_y += scroll_step
        driver.execute_script(f"window.scrollTo(0, {current_y});")
        time.sleep(pause_s)

        # After we scroll, see if the total height changed (new content appended).
        new_height = driver.execute_script("return document.body.scrollHeight")
        print(f"[DEBUG] scrolled_to={current_y}, innerH={inner_h}, pageY={page_y}, docH(old/new)={last_height}/{new_height}")

        # Only when height CHANGES do we parse and check the last review's date.
        if new_height != last_height:
            last_height = new_height

            # Parse only on height change
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            blocks = _find_review_blocks(soup)
            print(f"[DEBUG] height_changed -> review_blocks_found={len(blocks)}")

            if blocks:
                last_block = blocks[-1]
                # Look for a date like "Aug 15, 2025"
                m = re.search(r"([A-Za-z]{3}\s+\d{1,2},\s+\d{4})",
                              last_block.get_text(" ", strip=True))
                if m:
                    try:
                        last_dt = parse_date(m.group(1))
                        print(f"[DEBUG] last_review_date_after_height_change={last_dt}")
                        if last_dt < start_date:
                            print(f"[DEBUG] stop: last_review_date({last_dt}) < start_date({start_date})")
                            break  # condition (A)
                    except Exception as e:
                        print(f"[DEBUG] date-parse error: {e}")
        # If height did not change, we just continue stepping down until we hit bottom.
    print("[DEBUG] scroll_until_date completed.")

from typing import Optional

def extract_reviews(
    html: str,
    start_date: datetime.date,
    end_date: datetime.date,
    current_platform: str,
) -> list[dict]:
    """Parse reviews from HTML using flexible heuristics.

    Metacritic's DOM structure for user reviews has evolved, so we avoid
    relying on specific class names.  Instead, we locate review blocks
    using `_find_review_blocks` and then extract fields by searching
    within each block for dates, ratings, usernames and review text.

    Parameters
    ----------
    html : str
        The full HTML source of the user review page after all scrolling.
    start_date : datetime.date
        Start of the date range (inclusive).
    end_date : datetime.date
        End of the date range (inclusive).
    current_platform : str
        The name of the platform (e.g. "PlayStation 5") derived from the
        page URL or dropdown.  This value is assigned to each review
        extracted from the page, since Metacritic lists one platform per
        page.

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing `username`, `platform`,
        `rating`, `review_date` and `review_text` for a single review.
    """
    soup = BeautifulSoup(html, "html.parser")
    review_blocks = []
    # ------------------------------------------------------------------
    # Primary extraction: use explicit classes as seen in the user's
    # provided scraper.  Each review is contained in a div with the
    # following composite class.  If this yields results, it takes
    # precedence over the generic finder logic.
    primary_blocks = soup.find_all(
        'div', class_='c-siteReview g-bg-gray10 u-grid g-outer-spacing-bottom-large'
    )
    if primary_blocks:
        review_blocks = primary_blocks
    else:
        # Fall back to our generic finder if explicit classes are absent
        review_blocks = _find_review_blocks(soup)
    # Debug: report number of review blocks discovered
    try:
        print(f"[DEBUG] Found {len(review_blocks)} review blocks (primary={bool(primary_blocks)}) for platform '{current_platform}'")
    except Exception:
        pass
    extracted: list[dict] = []
    for index, block in enumerate(review_blocks, start=1):
        # DEBUG: print index of review being processed
        try:
            print(f"[DEBUG] Processing review block {index}/{len(review_blocks)}")
        except Exception:
            pass
        # Extract review text early; skip spoiler placeholders
        review_text = None
        quote_tag = block.find(
            'div', class_='c-siteReview_quote g-outer-spacing-bottom-small'
        )
        if quote_tag:
            review_text = quote_tag.get_text(strip=True)
        if review_text and '[SPOILER ALERT' in review_text:
            try:
                print(f"[DEBUG] Skipping review {index}: spoiler placeholder detected")
            except Exception:
                pass
            continue
        # Score (rating) is stored in a span with a specific Vue data attribute
        rating = None
        score_span = block.find('span', attrs={'data-v-e408cafe': True})
        if score_span and score_span.get_text(strip=True).isdigit():
            rating = int(score_span.get_text(strip=True))
        else:
            # fallback: look for any digit span
            sp = block.find('span', string=lambda s: s and s.strip().isdigit())
            if sp:
                try:
                    rating_val = int(sp.get_text(strip=True))
                    if 0 <= rating_val <= 10:
                        rating = rating_val
                except Exception:
                    rating = None
        if rating is None:
            try:
                print(f"[DEBUG] Skipping review {index}: rating not found or invalid")
            except Exception:
                pass
            continue
        # Username
        user_anchor = block.find(
            'a', class_='c-siteReviewHeader_username g-text-bold g-color-gray90'
        )
        if not user_anchor:
            # fallback pattern
            user_anchor = block.find('a', class_=re.compile('c-siteReviewHeader_username'))
        if user_anchor:
            username = user_anchor.get_text(strip=True)
        else:
            try:
                print(f"[DEBUG] Skipping review {index}: username not found")
            except Exception:
                pass
            continue
        # Date/time
        date_tag = block.find(
            'div', class_='c-siteReviewHeader_reviewDate g-color-gray80 u-text-uppercase'
        )
        if not date_tag:
            date_tag = block.find('div', class_=re.compile('c-siteReview_reviewDate'))
        if date_tag:
            date_str = date_tag.get_text(strip=True)
        else:
            # fallback to pattern search
            text = block.get_text(" ", strip=True)
            m = re.search(r"[A-Za-z]{3} \d{1,2}, \d{4}", text)
            date_str = m.group(0) if m else None
        if not date_str:
            try:
                print(f"[DEBUG] Skipping review {index}: date not found")
            except Exception:
                pass
            continue
        try:
            review_date = parse_date(date_str)
        except Exception:
            try:
                print(f"[DEBUG] Skipping review {index}: date parse error for '{date_str}'")
            except Exception:
                pass
            continue
        if review_date < start_date or review_date > end_date:
            try:
                print(f"[DEBUG] Skipping review {index}: date {review_date} outside range {start_date} - {end_date}")
            except Exception:
                pass
            continue
        # Platform
        platform_tag = block.find(
            'div', class_='c-siteReview_platform g-text-bold g-color-gray80 g-text-xsmall u-text-right u-text-uppercase'
        )
        if not platform_tag:
            platform_tag = block.find('div', class_=re.compile('c-siteReview_platform'))
        if platform_tag:
            platform = platform_tag.get_text(strip=True)
        else:
            platform = current_platform
        # Review text fallback: if review_text is None or trivial, try other tags
        if not review_text or review_text.strip() == '':
            # Attempt to capture text from generic quote classes
            qt = block.find('div', class_=re.compile('c-siteReview_quote'))
            if qt:
                review_text = qt.get_text(strip=True)
        if not review_text or not review_text.strip():
            # If still empty, scan for candidate strings as fallback
            block_strings = [s.strip() for s in block.stripped_strings if s.strip()]
            ignore_strings = {
                username,
                date_str,
                str(rating),
                'read more',
                'report',
                'positive',
                'mixed',
                'negative',
            }
            ignore_strings.add(current_platform)
            ignore_strings.add(current_platform.lower())
            candidates: list[str] = []
            for s in block_strings:
                sl = s.lower()
                if sl in (t.lower() for t in ignore_strings):
                    continue
                if re.fullmatch(r"\d+", s):
                    continue
                if re.match(r"[A-Za-z]{3} \d{1,2}, \d{4}", s):
                    continue
                if 'spoiler' in sl:
                    continue
                candidates.append(s)
            if candidates:
                review_text = max(candidates, key=len)
        # If no review text after all attempts, skip
        if not review_text or not review_text.strip():
            try:
                print(f"[DEBUG] Skipping review {index}: no review text found")
            except Exception:
                pass
            continue
        # Basic English filter
        if not is_english(review_text):
            try:
                print(f"[DEBUG] Skipping review {index}: non-English text detected")
            except Exception:
                pass
            continue
        # Append the extracted review
        try:
            print(f"[DEBUG] Accepted review {index}: username='{username}', date='{review_date}', rating={rating}, platform='{platform}', text_snippet='{review_text[:60]}...'")
        except Exception:
            pass
        extracted.append({
            'username': username,
            'platform': platform,
            'rating': rating,
            'review_date': review_date.strftime('%Y-%m-%d'),
            'review_text': review_text,
        })
    return extracted


def insert_reviews_to_db(db_path: str, table_name: str, reviews: list[dict]) -> None:
    """Insert review dictionaries into the specified SQLite database table.

    If the table does not exist, it is created with appropriate columns and a
    UNIQUE constraint to prevent duplicate rows.  Reviews are inserted using
    `INSERT OR IGNORE` so that any duplicates (determined by the unique
    constraint) are silently skipped.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    table_name : str
        Name of the table where reviews should be inserted.
    reviews : list[dict]
        A list of review dictionaries as returned by `extract_reviews`.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create table if it does not exist
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            platform TEXT,
            rating INTEGER,
            review_date TEXT,
            review_text TEXT,
            UNIQUE(username, platform, rating, review_date, review_text)
        )
        """
    )
    # Prepare insert statement
    insert_sql = (
        f"INSERT OR IGNORE INTO {table_name} "
        "(username, platform, rating, review_date, review_text) VALUES (?, ?, ?, ?, ?)"
    )
    # Convert review dicts to tuples
    values = [(
        review['username'],
        review['platform'],
        review['rating'],
        review['review_date'],
        review['review_text'],
    ) for review in reviews]
    cursor.executemany(insert_sql, values)
    conn.commit()
    conn.close()


def main() -> None:
    """Entry point for the script.  Prompts for user input and initiates scraping."""
    url = input(
        "Enter the Metacritic game URL (e.g. https://www.metacritic.com/game/elden-ring/): "
    ).strip()
    # Prompt for date range
    start_input = input(
        "Enter the start date (YYYY MM DD) or 'y' for default: "
    ).strip().lower()
    end_input = input(
        "Enter the end date (YYYY MM DD) or 'y' for default: "
    ).strip().lower()
    today = datetime.now().date()
    # Determine date range
    if start_input == 'y' and end_input == 'y':
        end_date = today
        start_date = end_date - timedelta(days=365 * 3)
    else:
        try:
            # Parse start date
            if start_input == 'y':
                start_date = today - timedelta(days=365 * 3)
            else:
                start_date = datetime.strptime(start_input, "%Y %m %d").date()
            # Parse end date
            if end_input == 'y':
                end_date = today
            else:
                end_date = datetime.strptime(end_input, "%Y %m %d").date()
            if start_date > end_date:
                raise ValueError("Start date must not be after end date.")
        except Exception:
            raise SystemExit("Invalid date format. Please use 'YYYY MM DD'.")
    # Extract game slug
    game_slug = slugify_game(url)
    # Build table name
    table_name = sanitize_table_name(
        f"{game_slug}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_metacritic"
    )
    # Database path
    db_path = os.path.join('Data_Extraction', 'Database', 'CS_Capstone.db')
    # Set up headless Chrome
    chrome_options = Options()
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # ----------------------------------------------------------------------
    # Locate or install ChromeDriver automatically
    #
    # Selenium requires an external ChromeDriver binary to control Chrome.  In
    # Selenium 4.6+ this can be handled by selenium‑manager if network access
    # is available, but in restricted environments Selenium may not be able to
    # download a driver.  To make this script robust across different
    # setups, we first try to find a system‑installed chromedriver using
    # shutil.which().  If one is found on the PATH, we use it.  Otherwise,
    # we fall back to using webdriver_manager to download the appropriate
    # driver.  If webdriver_manager is not installed, the user will need to
    # install it via pip or install chromedriver manually.
    import shutil
    from selenium.webdriver.chrome.service import Service
    # Determine if a chromedriver binary exists in the user's PATH
    chromedriver_path = shutil.which("chromedriver")
    driver = None  # type: ignore
    if chromedriver_path:
        # Found an existing driver; create a Service using that binary
        service = Service(chromedriver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
    else:
        # No local driver found; attempt to download via webdriver_manager with retries
        try:
            from webdriver_manager.chrome import ChromeDriverManager  # type: ignore
        except ImportError:
            raise SystemExit(
                "ChromeDriver could not be located automatically.\n"
                "Either install chromedriver manually and add it to your PATH,"
                " or install webdriver_manager via 'pip install webdriver-manager'."
            )
        # webdriver_manager is available; try to download the driver.  If the
        # network is temporarily unavailable, retry a few times before giving up.
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                driver_path = ChromeDriverManager().install()
                service = Service(driver_path)
                driver = webdriver.Chrome(service=service, options=chrome_options)
                break
            except Exception as err:  # catch any error during download/initialisation
                if attempt < max_attempts:
                    print(
                        f"Failed to download ChromeDriver (attempt {attempt}/{max_attempts}): {err}\n"
                        "Retrying in 15 seconds..."
                    )
                    time.sleep(15)
                    continue
                else:
                    raise SystemExit(
                        "ChromeDriver could not be downloaded automatically after multiple attempts.\n"
                        "Please install chromedriver manually and ensure it is on your PATH,"
                        " or check your network connection and try again."
                    )
    # Set a page load timeout to avoid hanging indefinitely on slow connections
    try:
        driver.set_page_load_timeout(120)
    except Exception:
        pass
    # Load the base user reviews page (without specifying platform) to extract platform codes
    base_reviews_url = url.rstrip('/') + '/user-reviews/'
    print(f"Opening {base_reviews_url} to discover platforms...")
    platform_codes: list[str] = []
    for attempt in range(1, 4):
        try:
            driver.get(base_reviews_url)
            print(f"[DEBUG] Loaded base user reviews page on attempt {attempt}")
            time.sleep(3)
            platform_codes = extract_platform_codes(driver.page_source)
            break
        except Exception as e:
            print(f"Failed to load {base_reviews_url} (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(5)
                continue
            else:
                print("Could not load base reviews page after multiple attempts; aborting.")
    if not platform_codes:
        # If we can't find platform codes, default to a generic list
        platform_codes = ['pc', 'ps5', 'xbox-series-x', 'ps4', 'xbox-one']
    print(f"[DEBUG] Found platforms: {platform_codes}")
    all_reviews: list[dict] = []
    # Human-readable names for common platform codes; fallback transforms hyphens
    platform_map = {
        'pc': 'PC',
        'playstation-5': 'PlayStation 5', 'ps5': 'PlayStation 5',
        'playstation-4': 'PlayStation 4', 'ps4': 'PlayStation 4',
        'xbox-series-x': 'Xbox Series X', 'xsx': 'Xbox Series X',
        'xbox-one': 'Xbox One', 'xb1': 'Xbox One',
    }
    for code in platform_codes:
        platform_url = base_reviews_url + f"?platform={code}"
        print(f"Processing platform '{code}' from {platform_url}")
        loaded = False
        for attempt in range(1, 4):
            try:
                driver.get(platform_url)
                print(f"[DEBUG] Loaded platform page {platform_url} on attempt {attempt}")
                # Wait for page to render
                time.sleep(3)
                loaded = True
                break
            except Exception as e:
                print(f"Failed to load {platform_url} (attempt {attempt}/3): {e}")
                if attempt < 3:
                    time.sleep(5)
                    continue
        if not loaded:
            print(f"Skipping platform {code} due to repeated load failures.")
            continue
        try:
            # Scroll and load all reviews within the date range
            scroll_until_date(driver, start_date)
        except Exception as e:
            print(f"Error while scrolling platform {code}: {e}")
        # Extract reviews from the loaded page
        platform_html = driver.page_source
        # Derive a user-friendly platform name
        platform_name = platform_map.get(code.lower(), code.replace('-', ' ').title())
        try:
            reviews = extract_reviews(platform_html, start_date, end_date, platform_name)
            print(f"Extracted {len(reviews)} reviews for platform '{platform_name}'")
            all_reviews.extend(reviews)
        except Exception as e:
            print(f"Failed to extract reviews from platform {code}: {e}")
            continue
    # Close the browser
    driver.quit()
    # Remove potential duplicates before inserting into database
    # Convert list of dicts to a dict keyed by unique tuple to deduplicate
    unique_map = {}
    for review in all_reviews:
        key = (
            review['username'],
            review['platform'],
            review['rating'],
            review['review_date'],
            review['review_text'],
        )
        unique_map[key] = review
    deduped_reviews = list(unique_map.values())
    # Insert into SQLite
    print(f"[DEBUG] Inserting {len(deduped_reviews)} unique reviews into database...")
    insert_reviews_to_db(db_path, table_name, deduped_reviews)
    print(f"Finished. Data stored in table '{table_name}' at '{db_path}'.")


if __name__ == '__main__':
    main()