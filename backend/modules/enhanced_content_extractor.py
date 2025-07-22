# modules/enhanced_content_extractor.py
from newspaper import Article, ArticleException
from bs4 import BeautifulSoup
import requests
from typing import Dict, List, Optional
import re
import logging
from urllib.parse import urlparse
import whois
from datetime import datetime
import time

class EnhancedContentExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Configure timeouts for better reliability
        self.timeout = 15  # 15 seconds timeout
        self.max_retries = 3
        
    def extract_comprehensive_content(self, url: str) -> Dict:
        """Extract comprehensive content from URL with enhanced metadata."""
        for attempt in range(self.max_retries):
            try:
                # Basic article extraction with timeout configuration
                article = Article(url)
                
                # Configure article with custom settings
                article.config.browser_user_agent = self.headers['User-Agent']
                article.config.request_timeout = self.timeout
                article.config.fetch_images = False  # Skip images to speed up extraction
                article.config.memoize_articles = False  # Don't cache to avoid stale data
                article.config.number_threads = 1  # Single thread to avoid conflicts
                
                # Download with retry logic
                download_success = False
                for download_attempt in range(2):  # Try download twice
                    try:
                        article.download()
                        download_success = True
                        break
                    except Exception as download_error:
                        if download_attempt < 1:  # If not the last attempt
                            logging.warning(f"Download attempt {download_attempt + 1} failed for {url}: {download_error}")
                            time.sleep(1)  # Wait 1 second before retry
                            continue
                        else:
                            raise download_error
                
                if not download_success:
                    raise ArticleException("Failed to download article after multiple attempts")
                
                article.parse()
                
                # Enhanced content extraction
                soup = BeautifulSoup(article.html, 'html.parser')
                
                extracted_data = {
                    'title': article.title,
                    'text': article.text,
                    'authors': article.authors,
                    'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                    'source_url': url,
                    'meta_description': self._extract_meta_description(soup),
                    'keywords': article.keywords,
                    'summary': article.summary,
                    'atomic_claims': self._extract_atomic_claims(article.text),
                    'domain_info': self._analyze_domain(url),
                    'content_quality_metrics': self._assess_content_quality(article.text),
                    'publication_indicators': self._extract_publication_indicators(soup),
                    'social_signals': self._extract_social_signals(soup)
                }
                
                return extracted_data
                
            except ArticleException as e:
                logging.error(f"Article extraction failed for {url} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # Try fallback method as last resort
                    logging.info(f"Trying fallback extraction method for {url}")
                    fallback_result = self._fallback_extraction(url)
                    if fallback_result and "error" not in fallback_result:
                        return fallback_result
                    return {"error": f"Article extraction failed after {self.max_retries} attempts: {str(e)}"}
            except Exception as e:
                logging.error(f"Unexpected error extracting content from {url} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # Try fallback method as last resort
                    logging.info(f"Trying fallback extraction method for {url}")
                    fallback_result = self._fallback_extraction(url)
                    if fallback_result and "error" not in fallback_result:
                        return fallback_result
                    return {"error": f"Unexpected error after {self.max_retries} attempts: {str(e)}"}
        
        return {"error": "Failed to extract content after all retry attempts"}
    
    def _fallback_extraction(self, url: str) -> Dict:
        """Fallback content extraction using direct requests when newspaper3k fails."""
        try:
            # Use requests with timeout
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic content
            title = self._extract_title_fallback(soup)
            text = self._extract_text_fallback(soup)
            
            if not text or len(text.strip()) < 50:
                return {"error": "Insufficient content extracted via fallback method"}
            
            extracted_data = {
                'title': title,
                'text': text,
                'authors': self._extract_authors_fallback(soup),
                'publish_date': self._extract_date_fallback(soup),
                'source_url': url,
                'meta_description': self._extract_meta_description(soup),
                'keywords': [],
                'summary': text[:200] + "..." if len(text) > 200 else text,
                'atomic_claims': self._extract_atomic_claims(text),
                'domain_info': self._analyze_domain(url),
                'content_quality_metrics': self._assess_content_quality(text),
                'publication_indicators': self._extract_publication_indicators(soup),
                'social_signals': self._extract_social_signals(soup)
            }
            
            return extracted_data
            
        except Exception as e:
            logging.error(f"Fallback extraction failed for {url}: {e}")
            return {"error": f"Fallback extraction failed: {str(e)}"}
    
    def _extract_title_fallback(self, soup: BeautifulSoup) -> str:
        """Extract title using fallback method."""
        # Try multiple selectors for title
        title_selectors = [
            'h1',
            'title',
            '[property="og:title"]',
            '[name="twitter:title"]',
            '.article-title',
            '.post-title',
            '.entry-title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title and len(title) > 5:
                    return title
        
        # Fallback to page title
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else "Untitled"
    
    def _extract_text_fallback(self, soup: BeautifulSoup) -> str:
        """Extract main text content using fallback method."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try to find main content area
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.main-content'
        ]
        
        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            # Fallback to body
            content_element = soup.find('body')
        
        if content_element:
            # Extract text and clean it
            text = content_element.get_text()
            # Clean up whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return ' '.join(lines)
        
        return ""
    
    def _extract_authors_fallback(self, soup: BeautifulSoup) -> List[str]:
        """Extract authors using fallback method."""
        author_selectors = [
            '.author',
            '.byline',
            '[rel="author"]',
            '[class*="author"]',
            '[class*="byline"]'
        ]
        
        authors = []
        for selector in author_selectors:
            elements = soup.select(selector)
            for element in elements:
                author_text = element.get_text().strip()
                if author_text and len(author_text) > 2:
                    authors.append(author_text)
        
        return list(set(authors))  # Remove duplicates
    
    def _extract_date_fallback(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date using fallback method."""
        date_selectors = [
            'time',
            '.date',
            '.published',
            '[property="article:published_time"]',
            '[name="publish_date"]'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                # Try to get datetime attribute first
                datetime_attr = element.get('datetime')
                if datetime_attr:
                    return datetime_attr
                
                # Fallback to text content
                date_text = element.get_text().strip()
                if date_text:
                    return date_text
        
        return None
    
    def _extract_atomic_claims(self, text: str) -> List[str]:
        """Extract atomic, verifiable claims from text with improved filtering."""
        # Clean the text first - remove HTML artifacts and metadata
        cleaned_text = self._clean_text_for_claim_extraction(text)
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)
        atomic_claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Skip sentences that are likely metadata, captions, or navigation
            if self._is_likely_metadata(sentence):
                continue
                
            # Check if sentence contains factual indicators
            factual_indicators = [
                r'\b\d{4}\b',  # Years
                r'\b\d+(\.\d+)?%\b',  # Percentages
                r'\b\d+(\,\d+)*\b',  # Large numbers
                r'\b(according to|study shows|research indicates|data reveals)\b',
                r'\b(CEO|president|director|minister|mayor)\b',
                r'\b(founded|established|created|launched)\b',
                r'\b(won|lost|defeated|beat|victory|championship|final)\b',  # Sports outcomes
                r'\b(announced|confirmed|reported|stated|said)\b',  # Reported events
                r'\b(World Cup|Olympics|championship|tournament)\b'  # Major events
            ]
            
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_indicators):
                # Ensure claim is contextually independent and clean
                contextualized_claim = self._contextualize_claim(sentence, cleaned_text)
                if self._is_valid_claim(contextualized_claim):
                    atomic_claims.append(contextualized_claim)
        
        # If no factual claims found, extract the main topic as a claim
        if not atomic_claims and cleaned_text:
            main_topic = self._extract_main_topic(cleaned_text)
            if main_topic:
                atomic_claims.append(main_topic)
        
        return atomic_claims[:5]  # Limit to top 5 most important claims
    
    def _clean_text_for_claim_extraction(self, text: str) -> str:
        """Clean text by removing HTML artifacts, metadata, and navigation elements."""
        # Remove common HTML artifacts
        text = re.sub(r'toggle caption', '', text, flags=re.IGNORECASE)
        text = re.sub(r'enlarge this image', '', text, flags=re.IGNORECASE)
        text = re.sub(r'via Getty Images', '', text, flags=re.IGNORECASE)
        text = re.sub(r'via AP', '', text, flags=re.IGNORECASE)
        text = re.sub(r'via Reuters', '', text, flags=re.IGNORECASE)
        
        # Remove image captions and photo credits
        text = re.sub(r'[A-Z][a-z]+/[A-Z][a-z]+', '', text)  # Remove photographer names like "David Ramos/FIFA"
        
        # Remove navigation and UI elements
        text = re.sub(r'\b(menu|navigation|search|login|sign up|subscribe)\b', '', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _is_likely_metadata(self, sentence: str) -> bool:
        """Check if a sentence is likely metadata, navigation, or non-factual content."""
        metadata_indicators = [
            r'toggle caption',
            r'enlarge this image',
            r'via Getty Images',
            r'via AP',
            r'via Reuters',
            r'click here',
            r'read more',
            r'share this',
            r'follow us',
            r'subscribe',
            r'newsletter',
            r'cookie policy',
            r'privacy policy',
            r'terms of service',
            r'contact us',
            r'about us',
            r'advertisement',
            r'sponsored',
            r'promoted'
        ]
        
        sentence_lower = sentence.lower()
        return any(re.search(pattern, sentence_lower) for pattern in metadata_indicators)
    
    def _is_valid_claim(self, claim: str) -> bool:
        """Check if a claim is valid and worth fact-checking."""
        # Must have minimum length
        if len(claim.strip()) < 30:
            return False
            
        # Must not be mostly metadata
        if self._is_likely_metadata(claim):
            return False
            
        # Must contain some factual content (numbers, names, dates, etc.)
        factual_elements = [
            r'\b\d{4}\b',  # Years
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper names
            r'\b(World Cup|Olympics|championship|tournament|final)\b',
            r'\b(won|lost|defeated|beat|victory)\b',
            r'\b(announced|confirmed|reported|stated)\b'
        ]
        
        return any(re.search(pattern, claim, re.IGNORECASE) for pattern in factual_elements)
    
    def _extract_main_topic(self, text: str) -> str:
        """Extract the main topic from text when no specific claims are found."""
        # Look for the most prominent factual statement
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 50:  # Skip very short sentences
                continue
                
            # Look for sentences with key factual elements
            if (re.search(r'\b(World Cup|championship|final)\b', sentence, re.IGNORECASE) and
                re.search(r'\b(Argentina|France|won|beat|defeated)\b', sentence, re.IGNORECASE)):
                return sentence[:200]  # Limit length
        
        # If no specific match, return the first substantial sentence
        for sentence in sentences:
            if len(sentence.strip()) > 50:
                return sentence.strip()[:200]
        
        return ""
    
    def _contextualize_claim(self, claim: str, full_text: str) -> str:
        """Add context to make claims independent."""
        # Simple contextualization - in production, use more sophisticated NLP
        pronouns = ['he', 'she', 'it', 'they', 'this', 'that', 'these', 'those']
        
        for pronoun in pronouns:
            if pronoun.lower() in claim.lower():
                # Find the most recent noun that could replace the pronoun
                # This is a simplified approach
                sentences_before = full_text[:full_text.find(claim)].split('.')
                for sentence in reversed(sentences_before[-3:]):  # Check last 3 sentences
                    nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
                    if nouns:
                        claim = claim.replace(pronoun, nouns[-1], 1)
                        break
        
        return claim
    
    def _analyze_domain(self, url: str) -> Dict:
        """Analyze domain characteristics for credibility assessment."""
        domain = urlparse(url).netloc.replace('www.', '')
        
        domain_info = {
            'domain': domain,
            'age_days': None,
            'registrar': None,
            'country': None,
            'is_news_domain': self._is_news_domain(domain),
            'is_academic_domain': domain.endswith('.edu') or domain.endswith('.ac.uk'),
            'is_government_domain': domain.endswith('.gov') or domain.endswith('.gov.uk')
        }
        
        try:
            whois_info = whois.whois(domain)
            if whois_info.creation_date:
                creation_date = whois_info.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                age = datetime.now() - creation_date
                domain_info['age_days'] = age.days
            
            domain_info['registrar'] = whois_info.registrar
            domain_info['country'] = whois_info.country
        except Exception as e:
            logging.warning(f"WHOIS lookup failed for {domain}: {e}")
        
        return domain_info
    
    def _is_news_domain(self, domain: str) -> bool:
        """Check if domain is a known news source."""
        news_indicators = [
            'news', 'times', 'post', 'herald', 'guardian', 'telegraph',
            'journal', 'gazette', 'tribune', 'daily', 'weekly', 'bbc',
            'cnn', 'reuters', 'ap.org', 'npr'
        ]
        return any(indicator in domain.lower() for indicator in news_indicators)
    
    def _assess_content_quality(self, text: str) -> Dict:
        """Assess various quality metrics of the content."""
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Check for quality indicators
        has_quotes = '"' in text or "'" in text
        has_sources = bool(re.search(r'\b(source|according to|study|research)\b', text, re.IGNORECASE))
        has_numbers = bool(re.search(r'\b\d+\b', text))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'has_quotes': has_quotes,
            'has_sources': has_sources,
            'has_numbers': has_numbers,
            'quality_score': self._calculate_quality_score(
                word_count, avg_sentence_length, has_quotes, has_sources, has_numbers
            )
        }
    
    def _calculate_quality_score(self, word_count: int, avg_sentence_length: float, 
                               has_quotes: bool, has_sources: bool, has_numbers: bool) -> float:
        """Calculate overall content quality score."""
        score = 0.0
        
        # Word count score (normalized)
        if word_count >= 500:
            score += 0.3
        elif word_count >= 200:
            score += 0.2
        elif word_count >= 100:
            score += 0.1
        
        # Sentence length score (optimal range 15-25 words)
        if 15 <= avg_sentence_length <= 25:
            score += 0.2
        elif 10 <= avg_sentence_length <= 30:
            score += 0.1
        
        # Content richness
        if has_quotes:
            score += 0.15
        if has_sources:
            score += 0.25
        if has_numbers:
            score += 0.1
        
        return min(1.0, score)  # Cap at 1.0
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract meta description from HTML."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content') if meta_desc else None
    
    def _extract_publication_indicators(self, soup: BeautifulSoup) -> Dict:
        """Extract publication-related indicators."""
        return {
            'has_byline': bool(soup.find(class_=re.compile('author|byline', re.I))),
            'has_dateline': bool(soup.find('time') or soup.find(class_=re.compile('date|time', re.I))),
            'has_social_share': bool(soup.find(class_=re.compile('share|social', re.I))),
            'has_comments': bool(soup.find(class_=re.compile('comment', re.I)))
        }
    
    def _extract_social_signals(self, soup: BeautifulSoup) -> Dict:
        """Extract social media signals if available."""
        return {
            'facebook_shares': self._extract_social_count(soup, 'facebook'),
            'twitter_shares': self._extract_social_count(soup, 'twitter'),
            'linkedin_shares': self._extract_social_count(soup, 'linkedin')
        }
    
    def _extract_social_count(self, soup: BeautifulSoup, platform: str) -> Optional[int]:
        """Extract social share count for a platform."""
        # This is a simplified implementation - social counts are often loaded via JavaScript
        elements = soup.find_all(class_=re.compile(platform, re.I))
        for element in elements:
            count_match = re.search(r'\d+', element.get_text())
            if count_match:
                return int(count_match.group())
        return None
