"""
fetcher.py
---------------

This script automates the process of harvesting the opening post from each
discussion thread listed on a forum index.  It was designed around the
Larian Studios forums, but the scraping logic is generic enough to handle
other forums powered by UBB.threads as well.  The tool will prompt for
the number of pages to crawl, iterate through each index page, extract
unique thread identifiers, fetch each thread and record the author,
timestamp, title and body of the first post into an SQLite database.

Key features
```
* Supports scraping either remote URLs (via HTTP/HTTPS) or local HTML
  files.  When a URL begins with ``file://`` or points to a path on
  disk, the scraper reads directly from the filesystem instead of
  making a network request.

* Robust thread discovery.  On a forum index page, threads are
  identified by the numeric ``Number`` parameter in links pointing to
  ``ubbthreads.php?ubb=showflat``.  Duplicates arising from multiple
  page links or anchors are deduplicated by thread number.

* Flexible page navigation.  The base index URL is amended with
  ``page`` query parameters to navigate subsequent pages.  If the
  provided URL lacks a ``page`` parameter, one is added.

* Careful first‑post extraction.  Within a thread page, the script
  isolates the subject row that does **not** start with ``"Re:"`` to
  capture the original thread title and timestamp.  The username of
  the author is read from the first occurrence of ``span.username``.
  The body of the post is taken from the ``div`` with ``id='body0'``.

* SQLite storage.  Harvested posts are stored in a table named
  ``forum_posts`` in the database located at
  ``Data_Extraction/Database/CS_Capstone.db``.  If the table does not
  already exist it will be created on the fly.  Each record includes
  the thread number, thread URL, page URL, title, author, post date,
  post time and content.

Usage example
-------------

```
$ python fetcher.py
Enter the forum index URL (page 1): file:///home/oai/share/forums.larian.com__ubbthreads.php__20250902_100501.html
How many pages would you like to fetch? 1
Scraping page 1: file:///home/oai/share/forums.larian.com__ubbthreads.php__20250902_100501.html
  Found 37 threads on this page
  Processing 37 threads...
  Inserted 1/37 posts into the database
  ...
Finished scraping 37 posts across 1 pages.
```

Note
----
The live forums at ``forums.larian.com`` require a logged‑in session to
access certain pages and may return HTTP 403 responses when scraped
anonymously.  For local testing you can pass paths to downloaded
HTML files prefaced with ``file://`` or directly as plain paths.  The
scraper will automatically detect and handle both cases.
"""

import os
import re
import sqlite3
import sys
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, urljoin

try:
    # ``requests`` is used for HTTP/HTTPS traffic.  It is optional if
    # scraping local files only.
    import requests
except ImportError:
    requests = None  # type: ignore

from bs4 import BeautifulSoup

# Path to the SQLite database.  The directory will be created if it
# does not exist.  Feel free to adjust this constant as needed.
DB_PATH = os.path.join("Data_Extraction", "Database", "Raw_Reviews.db")

TABLE_NAME = "Baldurs_gate_official_forum_posts"


def ensure_db() -> sqlite3.Connection:
    """Ensure that the database and table exist and return a connection.

    Returns
    -------
    sqlite3.Connection
        A connection object to the SQLite database.
    """
    # Create directories if necessary
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_number TEXT,
            thread_url TEXT,
            page_url TEXT,
            title TEXT,
            author TEXT,
            post_date TEXT,
            post_time TEXT,
            content TEXT
        )
        """
    )
    conn.commit()
    return conn


def get_html(url: str) -> bytes:
    """Retrieve raw HTML from a URL or local path.

    Parameters
    ----------
    url : str
        A URL beginning with ``http``, ``https`` or ``file://`` or a
        filesystem path.

    Returns
    -------
    bytes
        The raw HTML contents of the resource.

    Raises
    ------
    RuntimeError
        If fetching fails or the URL scheme is unsupported.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme in ("http", "https"):
        if requests is None:
            raise RuntimeError("The requests library is required for HTTP fetching")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch {url}: {exc}")
    elif scheme == "file":
        # Strip the leading "file://" and open the file directly
        path = parsed.path
        if not os.path.isfile(path):
            raise RuntimeError(f"File not found: {path}")
        with open(path, "rb") as f:
            return f.read()
    else:
        # If there is no scheme, treat it as a filesystem path
        if os.path.isfile(url):
            with open(url, "rb") as f:
                return f.read()
        raise RuntimeError(f"Unsupported URL or missing file: {url}")


def parse_forum_page(html: bytes) -> set[str]:
    """Extract unique thread numbers from a forum index page.

    The function looks for anchor tags containing the query parameter
    ``Number=``.  Only the numeric portion is retained.  Duplicate
    entries are automatically removed.

    Parameters
    ----------
    html : bytes
        Raw HTML of the forum index page.

    Returns
    -------
    set[str]
        A set of thread identifiers (as strings).
    """
    soup = BeautifulSoup(html, "html.parser")
    thread_numbers: set[str] = set()

    # The forum index renders each thread in a table row whose
    # ``<td>`` element includes the class "topicsubject".  Within this
    # cell the first ``<a>`` tag (without the ``pagenav`` or
    # ``pagenavall`` class) points to the thread itself.  Subsequent
    # links with class ``pagenav``/``pagenavall`` are simply page
    # navigation squares and should be ignored.  Limiting our search
    # to these cells avoids picking up numbers from unrelated areas of
    # the page (such as “Last Post” links).
    for td in soup.find_all("td", class_=lambda c: c and "topicsubject" in c):
        main_link = None
        for a in td.find_all("a", href=True):
            cls = a.get("class") or []
            # Normalise class into a list
            if isinstance(cls, str):
                cls = [cls]
            # Skip page navigation links (little square page numbers)
            if any("pagenav" in c for c in cls):
                continue
            main_link = a
            break
        if main_link:
            href = main_link["href"]
            m = re.search(r"Number=(\d+)", href)
            if m:
                thread_numbers.add(m.group(1))

    return thread_numbers


def parse_thread_page(html: bytes) -> dict:
    """Extract the first post information from a thread page.

    The function assumes that the thread is powered by UBB.threads and
    uses the following heuristics:

    * The page ``<title>`` contains the human‑readable thread title
      followed by a separator (``" - "``) and site branding.  The
      portion before the separator is returned as the title.
    * Each post is wrapped in a ``td`` element with class ``subjecttable``.
      The first of these where the embedded anchor text does not start
      with ``"Re: "`` is considered to represent the original post.  The
      posting date and time appear in sibling ``<span>`` elements with
      classes ``date`` and ``time``.
    * The username of the original poster appears in the first
      ``<span>`` with class ``username`` on the page.
    * The body of the original post lives in a ``<div>`` with id
      ``"body0"``.  Line breaks are preserved using newline
      separators.

    Parameters
    ----------
    html : bytes
        Raw HTML of the thread page.

    Returns
    -------
    dict
        A dictionary with keys ``title``, ``author``, ``post_date``,
        ``post_time`` and ``content``.  Missing values are returned
        as empty strings.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title from the <title> element
    title = ""
    if soup.title and soup.title.get_text():
        parts = soup.title.get_text().split(" - ")
        if parts:
            title = parts[0].strip()

    # Author name
    author = ""
    username_span = soup.find("span", class_="username")
    if username_span:
        author = username_span.get_text(strip=True)

    # Locate the subject table for the original post to capture its
    # timestamp.  Skip entries starting with "Re:" as they represent
    # replies.
    post_date = ""
    post_time = ""
    for td in soup.find_all("td", class_="subjecttable"):
        trunc = td.find("div", class_="truncate")
        if not trunc:
            continue
        anchor = trunc.find("a")
        if not anchor:
            continue
        title_text = anchor.get_text(strip=True)
        if title_text.startswith("Re:"):
            continue
        # Extract date and time from sibling spans within this table cell
        date_span = td.find("span", class_="date")
        time_span = td.find("span", class_="time")
        if date_span:
            post_date = date_span.get_text(strip=True)
        if time_span:
            post_time = time_span.get_text(strip=True)
        break

    # Body content of the original post
    content = ""
    body_div = soup.find("div", id="body0")
    if body_div:
        content = body_div.get_text(separator="\n").strip()

    return {
        "title": title,
        "author": author,
        "post_date": post_date,
        "post_time": post_time,
        "content": content,
    }


def update_query_parameter(url: str, param: str, value: str) -> str:
    """Return a new URL with the specified query parameter updated.

    If the parameter does not exist it will be added.

    Parameters
    ----------
    url : str
        The base URL to modify.
    param : str
        The name of the query parameter to update.
    value : str
        The value to set.

    Returns
    -------
    str
        The modified URL.
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query[param] = [value]
    new_query = urlencode(query, doseq=True)
    new_parts = list(parsed)
    new_parts[4] = new_query
    return urlunparse(new_parts)


def scrape_forum(forum_url: str, num_pages: int) -> None:
    """Scrape a specified number of forum index pages and store posts.

    Parameters
    ----------
    forum_url : str
        The URL or file path to the first page of the forum index.
    num_pages : int
        How many pages to scrape starting from page 1.

    Notes
    -----
    The function will iterate through pages 1 through ``num_pages`` by
    updating or appending a ``page`` query parameter.  If scraping a
    local file, only the first page will be read repeatedly as there
    are no subsequent files to fetch.
    """
    conn = ensure_db()
    cursor = conn.cursor()
    base_parsed = urlparse(forum_url)

    for page in range(1, num_pages + 1):
        # Compute page URL
        if base_parsed.scheme in ("file", "") and not base_parsed.netloc:
            # Local file: always reuse the same file, since there is no
            # canonical "page" concept when reading a single HTML file.
            current_url = forum_url
        else:
            # Remote URL: update or add page parameter
            current_url = update_query_parameter(forum_url, "page", str(page))

        print(f"Scraping page {page}: {current_url}")

        try:
            html = get_html(current_url)
        except Exception as exc:
            print(f"  Failed to retrieve page {page}: {exc}")
            continue

        thread_numbers = parse_forum_page(html)
        if not thread_numbers:
            print("  No threads found on this page.  Stopping.")
            break
        print(f"  Found {len(thread_numbers)} threads on this page")

        # Determine base domain to build absolute URLs for threads
        if base_parsed.scheme in ("http", "https"):
            domain = f"{base_parsed.scheme}://{base_parsed.netloc}"
        else:
            domain = ""

        # Process each thread
        inserted = 0
        for num in sorted(thread_numbers):
            # Construct thread URL.  Use ``showflat`` endpoint with the
            # thread number.  Avoid specifying ``page`` so that the first
            # page (page 1) of the thread loads implicitly.  When a
            # domain is available, build an absolute URL; otherwise use
            # a relative URL which ``get_html`` will treat as a local
            # file if it exists.
            if domain:
                thread_path = f"/ubbthreads.php?ubb=showflat&Number={num}"
                thread_url = urljoin(domain, thread_path)
            else:
                thread_url = f"/ubbthreads.php?ubb=showflat&Number={num}"

            try:
                thread_html = get_html(thread_url)
            except Exception as exc:
                # If we cannot fetch the thread (e.g. remote 403) we
                # continue to the next one.
                print(f"    Skipping thread {num}: {exc}")
                continue

            post_data = parse_thread_page(thread_html)
            if not post_data.get("title"):
                # Skip if no content extracted
                continue

            cursor.execute(
                f"""
                INSERT INTO {TABLE_NAME} (thread_number, thread_url, page_url,
                    title, author, post_date, post_time, content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    num,
                    thread_url,
                    current_url,
                    post_data.get("title", ""),
                    post_data.get("author", ""),
                    post_data.get("post_date", ""),
                    post_data.get("post_time", ""),
                    post_data.get("content", ""),
                ),
            )
            inserted += 1
            if inserted % 10 == 0:
                conn.commit()
            print(f"    Inserted {inserted}/{len(thread_numbers)} posts", end="\r")
        conn.commit()
        print(f"  Finished page {page}, inserted {inserted} posts.")
    conn.close()


def main() -> None:
    """Prompt for input and kick off the scraping process."""
    try:
        forum_url = input("Enter the forum index URL (page 1): ").strip()
        if not forum_url:
            print("No URL provided.  Exiting.")
            return
        pages = input("How many pages would you like to fetch? ").strip()
        try:
            num_pages = int(pages)
        except ValueError:
            print(f"Invalid number of pages: {pages}")
            return
        scrape_forum(forum_url, num_pages)
        print("Finished scraping.")
    except KeyboardInterrupt:
        print("\nScraping cancelled by user.")


if __name__ == "__main__":
    main()