from newspaper import Article, ArticleException

def extract_content_from_url(url):
    """
    Downloads an article from a URL and extracts its main title and text.

    Args:
        url (str): The URL of the news article or blog post.

    Returns:
        A dictionary containing the extracted title and text, or an error.
    """
    try:
        # Create an Article object
        article = Article(url)

        # Download the HTML content
        article.download()

        # Parse the article to extract the main content
        article.parse()

        # Check if parsing was successful
        if not article.text:
            return {"error": "Failed to extract main content from the URL."}

        # Return the relevant data in a dictionary
        return {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
            "source_url": url
        }
    except ArticleException as e:
        print(f"Newspaper3k error for URL {url}: {e}")
        return {"error": f"Could not process the URL. It may be invalid or inaccessible."}
    except Exception as e:
        print(f"An unexpected error occurred for URL {url}: {e}")
        return {"error": "An unexpected error occurred during content extraction."}