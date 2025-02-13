import os
from typing import Dict, List, Optional, Any, TypedDict, Literal, NotRequired
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import json
from urllib.parse import urlparse, urljoin, unquote, urlunparse, quote
import logging
import csv
from pathlib import Path
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

def get_api_key(key_name: str) -> Optional[str]:
    """Get API key from environment or return None."""
    return os.getenv(key_name)

# API Keys will be set dynamically when needed
SCRAPINGANT_API_KEY: Optional[str] = None
EXA_API_KEY: Optional[str] = None

def set_api_keys(scrapingant_key: Optional[str] = None, exa_key: Optional[str] = None):
    """Set API keys globally."""
    global SCRAPINGANT_API_KEY, EXA_API_KEY
    
    # Set provided keys if they are not empty
    if scrapingant_key:
        SCRAPINGANT_API_KEY = scrapingant_key
        logger.info("Set ScrapingAnt API key from provided value")
    
    if exa_key:
        EXA_API_KEY = exa_key
        logger.info("Set Exa.ai API key from provided value")
    
    # If any key is still not set, try environment variables
    if not SCRAPINGANT_API_KEY:
        SCRAPINGANT_API_KEY = os.getenv("SCRAPINGANT_API_KEY", "")
        if SCRAPINGANT_API_KEY:
            logger.info("Set ScrapingAnt API key from environment")
    
    if not EXA_API_KEY:
        EXA_API_KEY = os.getenv("EXA_API_KEY", "")
        if EXA_API_KEY:
            logger.info("Set Exa.ai API key from environment")

def validate_api_keys():
    """Validate that required API keys are set."""
    global SCRAPINGANT_API_KEY, EXA_API_KEY
    
    if not SCRAPINGANT_API_KEY or not EXA_API_KEY:
        # Try to get keys from environment first
        SCRAPINGANT_API_KEY = os.getenv("SCRAPINGANT_API_KEY", "")
        EXA_API_KEY = os.getenv("EXA_API_KEY", "")
    
    # If still not set, raise appropriate error
    if not SCRAPINGANT_API_KEY or not EXA_API_KEY:
        raise ValueError("API keys are required. Please set your API keys in the sidebar before proceeding.")

def get_required_env_var(var_name: str) -> str:
    """Get a required environment variable or raise an error with a helpful message."""
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {var_name}\n"
            f"Please set this in your .env file or environment variables.\n"
            f"You can copy .env.example to .env and fill in your values."
        )
    return value

@dataclass
class ApiUsage:
    timestamp: datetime
    api_name: str
    endpoint: str
    credits_used: int
    cost: float
    status: str
    response_time: float

@dataclass
class MediaAsset:
    type: str
    url: str
    alt: str = ""
    metadata: Dict = None

class CompanyAssets(TypedDict):
    logos: list[dict[Literal["url", "source", "alt"], str]]
    social_media: list[dict[Literal["platform", "url", "source"], str]]
    media: list[dict[Literal["type", "url", "source"], str]]

class CompanyMetadata(TypedDict):
    name: NotRequired[str]
    description: NotRequired[str]
    founded: NotRequired[str]
    employees: NotRequired[str]
    headquarters: NotRequired[str]
    sources: dict[str, list[str]]

class CompanyData(TypedDict):
    basic_info: dict[str, str]
    assets: CompanyAssets
    metadata: CompanyMetadata
    technologies: list[str]
    data_sources: list[str]

class ApiUsageTracker:
    def __init__(self, csv_file: str = "api_usage.csv"):
        self.csv_file = csv_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        if not Path(self.csv_file).exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'api_name', 'endpoint', 'credits_used', 
                    'cost', 'status', 'response_time'
                ])
                writer.writeheader()
    
    def log_usage(self, usage: ApiUsage):
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'api_name', 'endpoint', 'credits_used', 
                'cost', 'status', 'response_time'
            ])
            writer.writerow(usage.__dict__)

class ExaSearcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.exa.ai"  # Base URL
        self.usage_tracker = ApiUsageTracker()
    
    def query_exa(self, query: str, url: str) -> List[Dict[str, Any]]:
        """Query Exa.ai API with better error handling and domain validation."""
        start_time = datetime.now()
        
        try:
            # Extract and validate domain
            parsed_url = urlparse(url if url.startswith(('http://', 'https://')) else f'https://{url}')
            domain = parsed_url.netloc.lower()
            if not domain:
                raise ValueError(f"Invalid domain extracted from URL: {url}")
            
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Validate domain format using raw string for regex
            if not re.match(r'^[a-z0-9]+([-.][a-z0-9]+)*[.][a-z]{2,}$', domain):
                raise ValueError(f"Invalid domain format: {domain}")
            
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "x-api-key": self.api_key
            }
            
            # Prepare request body with validated domain
            payload = {
                "query": query,
                "numResults": 10,
                "includeDomains": [domain],
                "excludeDomains": [],
                "startPublishedDate": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                "endPublishedDate": datetime.now().strftime("%Y-%m-%d"),
                "useAutoprompt": True,
                "type": "keyword",
                "minWordCount": 100,
                "maxWordCount": 2000
            }
            
            response = requests.post(
                f"{self.base_url}/search",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, dict) or "results" not in data:
                raise ValueError(f"Unexpected response format from Exa.ai: {data}")
            
            self.usage_tracker.log_usage(ApiUsage(
                timestamp=datetime.now(),
                api_name="Exa.ai",
                endpoint="/search",
                credits_used=1,
                cost=0.0,
                status="success",
                response_time=(datetime.now() - start_time).total_seconds()
            ))
            
            return data.get("results", [])
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Exa.ai API request failed: {str(e)}"
            logger.error(error_msg)
            self.usage_tracker.log_usage(ApiUsage(
                timestamp=datetime.now(),
                api_name="Exa.ai",
                endpoint="/search",
                credits_used=0,
                cost=0.0,
                status="error",
                response_time=(datetime.now() - start_time).total_seconds()
            ))
            raise ValueError(error_msg)
            
        except ValueError as e:
            error_msg = f"Invalid input or response: {str(e)}"
            logger.error(error_msg)
            self.usage_tracker.log_usage(ApiUsage(
                timestamp=datetime.now(),
                api_name="Exa.ai",
                endpoint="/search",
                credits_used=0,
                cost=0.0,
                status="error",
                response_time=(datetime.now() - start_time).total_seconds()
            ))
            raise ValueError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error querying Exa.ai: {str(e)}"
            logger.error(error_msg)
            self.usage_tracker.log_usage(ApiUsage(
                timestamp=datetime.now(),
                api_name="Exa.ai",
                endpoint="/search",
                credits_used=0,
                cost=0.0,
                status="error",
                response_time=(datetime.now() - start_time).total_seconds()
            ))
            raise ValueError(error_msg)

    def extract_company_info(self, search_results: Dict) -> Optional[CompanyData]:
        """Extract company information from Exa.ai search results"""
        try:
            if not search_results or "results" not in search_results:
                return None
            
            # Combine information from multiple results
            results = search_results["results"]
            if not results:
                return None
            
            # Get the most relevant result
            best_result = results[0]
            
            # Extract company name from the title
            company_name = best_result.get("title", "").split("-")[0].strip()
            
            # Extract domain from the URL
            domain = urlparse(best_result.get("url", "")).netloc
            
            # Combine descriptions from multiple results
            descriptions = [r.get("text", "") for r in results if r.get("text")]
            description = descriptions[0] if descriptions else None
            
            return CompanyData(
                name=company_name,
                domain=domain,
                description=description,
                logo_url=None,  # Exa.ai doesn't provide logos
                social_links=[],  # Will be enriched by ScrapingAnt
                media_assets=[],  # Will be enriched by ScrapingAnt
                scrape_date=datetime.now(),
                product_lines=None,  # Will be populated by ScrapingAnt
                competitors=None,  # Will be populated by ScrapingAnt
                industry=None,  # Will be determined by ScrapingAnt
                key_metrics=None,  # Will be populated by ScrapingAnt
                technologies=None,  # Will be populated by ScrapingAnt
                # New fields for M&A and market research
                executives=None,  # Will be populated by ScrapingAnt
                funding_rounds=None,  # Will be populated by ScrapingAnt
                documents=None,  # Will be populated by ScrapingAnt
                videos=None,  # Will be populated by ScrapingAnt
                achievements=None,  # Will be populated by ScrapingAnt
                financial_metrics=None,  # Will be populated by ScrapingAnt
                market_position=None,  # Will be populated by ScrapingAnt
                growth_indicators=None,  # Will be populated by ScrapingAnt
                # New timeline field
                timeline_events=None,  # Will be populated by ScrapingAnt
                benchmarking_data=None,  # Will be populated by ScrapingAnt
                positioning_analysis=None,  # Will be populated by ScrapingAnt
                data_sources=None  # Will be populated by ScrapingAnt
            )
        except Exception as e:
            logger.error(f"Error extracting company info from Exa.ai results: {str(e)}")
            return None

class ScrapingAntScraper:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.base_url = "https://api.scrapingant.com/v2"
        self.usage_tracker = ApiUsageTracker()
        self.max_retries = max_retries
    
    def _scrape_with_fallback(self, url: str) -> tuple[str, list[str]]:
        """Scrape with multiple fallback methods tracking data sources"""
        sources = []
        try:
            # Primary scrape with ScrapingAnt
            html, api_source = self._scrape_via_api(url)
            sources.append(api_source)
            return html, sources
        except Exception as api_error:
            logger.warning(f"API scrape failed: {str(api_error)}")
            try:
                # Fallback to direct requests
                html, direct_source = self._scrape_direct(url)
                sources.append(direct_source)
                return html, sources
            except Exception as direct_error:
                logger.error(f"All scrape methods failed: {str(direct_error)}")
                return "", sources

    def _scrape_via_api(self, url: str) -> tuple[str, str]:
        """Scrape using ScrapingAnt API with proper source tracking"""
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL format: {url}")
        
        # Ensure URL is properly formatted and encoded
        clean_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path or '/',  # Ensure there's at least a forward slash
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment
        ))

        # Prepare headers
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json'
        }

        # Prepare query parameters
        params = {
            'url': clean_url,
            'x-api-key': self.api_key,
            'browser': 'true',
            'proxy_type': 'residential',
            'render_js': 'true',
            'return_text': 'true',
            'wait_for': 'networkidle0'
        }

        try:
            # Make GET request with query parameters
            response = requests.get(
                f"{self.base_url}/general",
                headers=headers,
                params=params,
                timeout=90
            )

            # Log response details for debugging
            logger.info(f"ScrapingAnt response status: {response.status_code}")
            logger.info(f"ScrapingAnt response headers: {response.headers}")
            logger.info(f"ScrapingAnt request URL: {response.request.url}")

            if response.status_code != 200:
                error_msg = f"API request failed with status code: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f". Error: {error_data['error']}"
                    elif 'detail' in error_data:
                        error_msg += f". Detail: {error_data['detail']}"
                except:
                    error_msg += f". Response: {response.text[:200]}"
                raise RuntimeError(error_msg)

            # Check content type from headers
            content_type = response.headers.get('Content-Type', '').lower()

            if 'application/json' in content_type:
                # Handle JSON response
                try:
                    response_data = response.json()
                    if "error" in response_data:
                        raise RuntimeError(f"ScrapingAnt API Error: {response_data['error']}")
                    
                    # Get content from JSON response
                    content = response_data.get("content")
                    if not content:
                        content = response_data.get("text")  # Try alternate field
                    if not content:
                        content = response_data.get("html")  # Try another alternate field
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {str(e)}")
                    logger.error(f"Response content: {response.content[:200]}")
                    raise ValueError(f"Invalid JSON response from ScrapingAnt: {str(e)}")
            else:
                # Handle HTML response directly
                content = response.text

            if not content:
                raise ValueError("No content returned from ScrapingAnt")

            return content, "scrapingant_api"

        except requests.exceptions.RequestException as e:
            logger.error(f"Request to ScrapingAnt failed: {str(e)}")
            raise RuntimeError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in ScrapingAnt request: {str(e)}")
            raise RuntimeError(f"Scraping failed: {str(e)}")

    def _scrape_direct(self, url: str) -> tuple[str, str]:
        """Direct scraping fallback with source tracking"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text, "direct_request"
        except requests.RequestException as e:
            raise RuntimeError(f"Direct request failed: {str(e)}")
        
    def scrape_page(self, url: str) -> Optional[str]:
        start_time = datetime.now()
        retry_count = 0
        last_error = None

        try:
            # Trim any extra whitespace
            url = url.strip()
            
            # Parse and validate URL
            parsed_url = urlparse(url)
            logger.info(f"Parsed URL: {parsed_url}")
            if not parsed_url.scheme:
                url = f"https://{url}"
                parsed_url = urlparse(url)
            if not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {url}")
            # If no path is provided, add a trailing slash
            if not parsed_url.path:
                url = f"{parsed_url.scheme}://{parsed_url.netloc}/"

            # First try with ScrapingAnt
            while retry_count < self.max_retries:
                try:
                    logger.info(f"Attempt {retry_count + 1}/{self.max_retries} to scrape {url}")
                    
                    content, source = self._scrape_via_api(url)
                    
                    if not content:
                        raise ValueError("Empty content received from scraping")

                    # Log success
                    self.usage_tracker.log_usage(ApiUsage(
                        timestamp=datetime.now(),
                        api_name="ScrapingAnt",
                        endpoint="/general",
                        credits_used=1,
                        cost=0.0,
                        status="success",
                        response_time=(datetime.now() - start_time).total_seconds()
                    ))

                    return content

                except (requests.exceptions.RequestException, ValueError, RuntimeError) as e:
                    last_error = str(e)
                    logger.error(f"ScrapingAnt Error (Attempt {retry_count + 1}): {last_error}")
                    retry_count += 1
                    
                    if retry_count < self.max_retries:
                        wait_time = min(2 ** retry_count, 30)  # Cap wait time at 30 seconds
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"All ScrapingAnt attempts failed, trying fallback method...")

            # If ScrapingAnt fails, try with regular requests as fallback
            try:
                logger.info("Attempting fallback scraping method...")
                session = requests.Session()
                session.max_redirects = 5
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                response = session.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                content = response.text
                if not content:
                    raise ValueError("Empty content received from fallback method")

                # Log fallback success
                self.usage_tracker.log_usage(ApiUsage(
                    timestamp=datetime.now(),
                    api_name="ScrapingAnt",
                    endpoint="/scrape",
                    credits_used=0,
                    cost=0.0,
                    status="fallback_success",
                    response_time=(datetime.now() - start_time).total_seconds()
                ))
                
                return content
            
            except Exception as e:
                logger.error(f"Fallback scraping failed: {str(e)}")
                self.usage_tracker.log_usage(ApiUsage(
                    timestamp=datetime.now(),
                    api_name="ScrapingAnt",
                    endpoint="/scrape",
                    credits_used=0,
                    cost=0.0,
                    status="error",
                    response_time=(datetime.now() - start_time).total_seconds()
                ))
                raise ValueError(f"Failed to scrape {url} after {self.max_retries} attempts and fallback. Last error: {last_error}")

        except Exception as e:
            logger.error(f"Unexpected error processing URL: {str(e)}")
            raise ValueError(f"Failed to scrape {url}: {str(e)}")

    def extract_product_lines(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract product line information from the website with enhanced detection"""
        product_lines = []
        
        # Common product section identifiers
        product_sections = soup.find_all(['section', 'div', 'ul', 'nav'], class_=lambda x: x and any(
            term in x.lower() for term in [
                'product', 'solution', 'service', 'offering', 'collection', 
                'category', 'featured', 'portfolio', 'showcase'
            ]
        ))
        
        # Also look for product sections by ID
        product_sections.extend(
            soup.find_all(id=lambda x: x and any(
                term in x.lower() for term in [
                    'product', 'solution', 'service', 'offering', 
                    'collection', 'category', 'featured'
                ]
            ))
        )
        
        # Look for menu items that might indicate product categories
        menu_items = soup.find_all(['a', 'li'], class_=lambda x: x and any(
            term in x.lower() for term in ['menu-item', 'nav-item', 'dropdown']
        ))
        
        for item in menu_items:
            href = item.get('href', '')
            text = item.get_text(strip=True)
            if any(term in href.lower() or term in text.lower() for term in ['product', 'solution', 'service']):
                parent = item.find_parent(['ul', 'nav', 'div'])
                if parent:
                    product_sections.append(parent)
        
        for section in product_sections:
            # Look for product cards/items
            product_items = section.find_all(['div', 'article', 'li', 'a'], class_=lambda x: x and any(
                term in x.lower() for term in ['item', 'card', 'product', 'solution', 'service']
            ))
            
            if not product_items:
                # If no specific items found, try to find headings that might indicate products
                product_items = section.find_all(['h2', 'h3', 'h4', 'h5'])
            
            for item in product_items:
                product_info = {
                    'name': '',
                    'description': '',
                    'category': '',
                    'features': [],
                    'image_url': None,
                    'product_url': None
                }
                
                # Extract name from heading or strong text
                name_elem = item.find(['h2', 'h3', 'h4', 'h5', 'strong']) or item
                product_info['name'] = name_elem.get_text(strip=True)
                
                # Skip if name is too generic or empty
                if not product_info['name'] or len(product_info['name']) < 3:
                    continue
                
                # Extract description
                desc_elem = item.find(['p', 'div'], class_=lambda x: x and any(
                    term in x.lower() for term in ['description', 'content', 'text', 'summary']
                ))
                if desc_elem:
                    product_info['description'] = desc_elem.get_text(strip=True)
                
                # Extract features
                feature_list = item.find(['ul', 'ol'])
                if feature_list:
                    features = feature_list.find_all('li')
                    product_info['features'] = [
                        feature.get_text(strip=True) for feature in features
                        if feature.get_text(strip=True)
                    ]
                
                # Extract image
                img = item.find('img')
                if img and img.get('src'):
                    src = img['src']
                    if not src.startswith(('http://', 'https://')):
                        # Convert relative URL to absolute
                        base_url = self.base_url
                        src = urljoin(base_url, src)
                    product_info['image_url'] = src
                
                # Extract product URL
                product_link = item.find('a') if not item.name == 'a' else item
                if product_link and product_link.get('href'):
                    href = product_link['href']
                    if not href.startswith(('http://', 'https://')):
                        href = urljoin(self.base_url, href)
                    product_info['product_url'] = href
                
                # Try to determine category based on context
                parent_text = ' '.join(p.get_text(strip=True).lower() for p in item.parents)
                category_hints = ['software', 'hardware', 'service', 'solution', 'platform', 'tool', 'app']
                for hint in category_hints:
                    if hint in parent_text:
                        product_info['category'] = hint.title()
                        break
                
                # Add to product lines if we have meaningful information
                if product_info['name'] and (
                    product_info['description'] or 
                    product_info['features'] or 
                    product_info['image_url'] or 
                    product_info['product_url']
                ):
                    product_lines.append(product_info)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_products = []
        for product in product_lines:
            if product['name'] not in seen:
                seen.add(product['name'])
                unique_products.append(product)
        
        return unique_products

    def extract_timeline_events(self, soup: BeautifulSoup, company_name: str, exa_searcher: Optional[ExaSearcher] = None) -> List[Dict[str, Any]]:
        """Extract timeline events from the website and news sources"""
        timeline_events = []
        
        # Extract events from the website
        event_sections = soup.find_all(['section', 'div'], class_=lambda x: x and any(
            term in x.lower() for term in ['timeline', 'history', 'milestones', 'journey', 'story']
        ))
        
        for section in event_sections:
            events = section.find_all(['div', 'article', 'li'], class_=lambda x: x and any(
                term in x.lower() for term in ['event', 'milestone', 'item', 'entry']
            ))
            
            for event in events:
                # Extract date
                date_elem = event.find(text=re.compile(r'\d{4}'))
                if not date_elem:
                    continue
                
                # Extract title
                title_elem = event.find(['h3', 'h4', 'strong', 'b'])
                title = title_elem.get_text(strip=True) if title_elem else None
                
                # Extract description
                desc_elem = event.find(['p', 'div'], class_=lambda x: x and 'description' in str(x).lower())
                description = desc_elem.get_text(strip=True) if desc_elem else None
                
                if title or description:
                    timeline_events.append({
                        'date': date_elem.strip(),
                        'title': title,
                        'description': description,
                        'type': 'website',
                        'source_url': None
                    })
        
        # If Exa.ai is available, get news events
        if exa_searcher:
            try:
                # Search for news about the company
                news_query = f"{company_name} (funding OR acquisition OR launch OR milestone OR achievement)"
                news_results = exa_searcher.query_exa(news_query, "")
                
                for result in news_results:
                    # Extract date from the result
                    date_match = re.search(r'\b\d{4}\b', result.get('snippet', ''))
                    if not date_match:
                        continue
                    
                    timeline_events.append({
                        'date': date_match.group(),
                        'title': result.get('title'),
                        'description': result.get('snippet'),
                        'type': 'news',
                        'source_url': result.get('url')
                    })
            except Exception as e:
                logger.warning(f"Failed to fetch news events: {str(e)}")
        
        # Sort events by date
        timeline_events.sort(key=lambda x: x['date'])
        
        return timeline_events

    def analyze_benchmarking_data(self, company_data: CompanyData, competitors: List[Dict[str, str]], exa_searcher: ExaSearcher) -> Dict[str, Any]:
        """Analyze and compare company metrics with competitors"""
        benchmarking_data = {
            'metrics_comparison': {},
            'strength_weakness': {},
            'market_trends': [],
            'competitive_advantages': []
        }
        
        try:
            # Collect metrics for comparison
            metrics_to_compare = [
                'revenue',
                'employee_count',
                'market_share',
                'funding',
                'product_count',
                'customer_base',
                'geographic_presence'
            ]
            
            # Initialize metrics comparison
            for metric in metrics_to_compare:
                benchmarking_data['metrics_comparison'][metric] = {
                    'company': None,
                    'competitors': {},
                    'industry_avg': None
                }
            
            # Extract company metrics
            company_metrics = {}
            for metric in metrics_to_compare:
                value = company_data.key_metrics.get(metric)
                if value:
                    company_metrics[metric] = self._normalize_metric(value)
            
            # Get competitor metrics
            for competitor in competitors[:5]:  # Focus on top 5 competitors
                try:
                    # Search for competitor metrics
                    query = f"{competitor['name']} {' OR '.join(metrics_to_compare)}"
                    results = exa_searcher.query_exa(query, "")
                    
                    competitor_metrics = {}
                    for result in results:
                        snippet = result.get('snippet', '').lower()
                        
                        # Look for metric mentions
                        for metric in metrics_to_compare:
                            if metric in snippet:
                                # Extract metric value using regex patterns
                                patterns = [
                                    rf"{metric}.*?(\d+(?:\.\d+)?(?:k|m|b|t)?)",
                                    rf"(\d+(?:\.\d+)?(?:k|m|b|t)?).*?{metric}"
                                ]
                                
                                for pattern in patterns:
                                    match = re.search(pattern, snippet, re.IGNORECASE)
                                    if match:
                                        value = self._normalize_metric(match.group(1))
                                        competitor_metrics[metric] = value
                                        break
                    
                    if competitor_metrics:
                        for metric, value in competitor_metrics.items():
                            benchmarking_data['metrics_comparison'][metric]['competitors'][competitor['name']] = value
                
                except Exception as e:
                    logger.warning(f"Failed to get metrics for {competitor['name']}: {str(e)}")
            
            # Calculate industry averages
            for metric in metrics_to_compare:
                values = [company_metrics.get(metric)]
                values.extend(
                    v for v in benchmarking_data['metrics_comparison'][metric]['competitors'].values()
                    if v is not None
                )
                
                if values:
                    avg = sum(v for v in values if v is not None) / len([v for v in values if v is not None])
                    benchmarking_data['metrics_comparison'][metric]['industry_avg'] = avg
            
            # Analyze strengths and weaknesses
            for metric in metrics_to_compare:
                company_value = company_metrics.get(metric)
                if company_value is not None:
                    industry_avg = benchmarking_data['metrics_comparison'][metric]['industry_avg']
                    if industry_avg is not None:
                        if company_value > industry_avg * 1.1:  # 10% above average
                            benchmarking_data['strength_weakness']['strengths'] = \
                                benchmarking_data['strength_weakness'].get('strengths', []) + [metric]
                        elif company_value < industry_avg * 0.9:  # 10% below average
                            benchmarking_data['strength_weakness']['weaknesses'] = \
                                benchmarking_data['strength_weakness'].get('weaknesses', []) + [metric]
            
            # Analyze market trends
            try:
                trend_query = f"{company_data.industry} industry trends 2024"
                trend_results = exa_searcher.query_exa(trend_query, "")
                
                for result in trend_results:
                    snippet = result.get('snippet', '')
                    if 'trend' in snippet.lower() or 'growing' in snippet.lower():
                        benchmarking_data['market_trends'].append({
                            'trend': snippet,
                            'source': result.get('url')
                        })
            except Exception as e:
                logger.warning(f"Failed to analyze market trends: {str(e)}")
            
            # Identify competitive advantages
            if company_data.market_position:
                advantages = company_data.market_position.get('advantages', [])
                benchmarking_data['competitive_advantages'].extend(advantages)
            
            return benchmarking_data
            
        except Exception as e:
            logger.error(f"Error analyzing benchmarking data: {str(e)}")
            return benchmarking_data

    def analyze_market_positioning(self, company_data: CompanyData, competitors: List[Dict[str, str]], exa_searcher: ExaSearcher) -> Dict[str, Any]:
        """Analyze market positioning relative to competitors"""
        positioning_analysis = {
            'market_segments': [],
            'positioning_map': {
                'dimensions': [],
                'company_position': {},
                'competitor_positions': {}
            },
            'differentiation_factors': [],
            'target_audience': [],
            'growth_opportunities': []
        }
        
        try:
            # Identify market segments
            if company_data.market_position:
                segments = []
                for key, value in company_data.market_position.items():
                    if 'segment' in key.lower() or 'market' in key.lower():
                        segments.append(value)
                positioning_analysis['market_segments'].extend(segments)
            
            # Create positioning map
            positioning_dimensions = [
                ('price', 'quality'),
                ('innovation', 'reliability'),
                ('customization', 'standardization'),
                ('local', 'global')
            ]
            
            # Select the most relevant dimension pair
            selected_dimension = positioning_dimensions[0]  # Default to first pair
            dimension_relevance = {}
            
            for dim_pair in positioning_dimensions:
                relevance_score = 0
                dim_query = f"{company_data.name} {dim_pair[0]} {dim_pair[1]}"
                try:
                    results = exa_searcher.query_exa(dim_query, "")
                    relevance_score = len(results)
                    dimension_relevance[dim_pair] = relevance_score
                except Exception:
                    continue
            
            if dimension_relevance:
                selected_dimension = max(dimension_relevance.items(), key=lambda x: x[1])[0]
            
            positioning_analysis['positioning_map']['dimensions'] = list(selected_dimension)
            
            # Analyze company position
            try:
                position_query = f"{company_data.name} {selected_dimension[0]} {selected_dimension[1]}"
                results = exa_searcher.query_exa(position_query, "")
                
                # Score position based on mentions and context
                dim1_score = 0
                dim2_score = 0
                
                for result in results:
                    snippet = result.get('snippet', '').lower()
                    
                    # Score first dimension
                    if selected_dimension[0] in snippet:
                        context_words = snippet.split()
                        idx = context_words.index(selected_dimension[0])
                        if idx > 0:
                            modifier = context_words[idx - 1]
                            if modifier in ['high', 'higher', 'highest']:
                                dim1_score += 1
                            elif modifier in ['low', 'lower', 'lowest']:
                                dim1_score -= 1
                    
                    # Score second dimension
                    if selected_dimension[1] in snippet:
                        context_words = snippet.split()
                        idx = context_words.index(selected_dimension[1])
                        if idx > 0:
                            modifier = context_words[idx - 1]
                            if modifier in ['high', 'higher', 'highest']:
                                dim2_score += 1
                            elif modifier in ['low', 'lower', 'lowest']:
                                dim2_score -= 1
                
                # Normalize scores to 0-100 range
                positioning_analysis['positioning_map']['company_position'] = {
                    selected_dimension[0]: min(max((dim1_score + 3) * 20, 0), 100),
                    selected_dimension[1]: min(max((dim2_score + 3) * 20, 0), 100)
                }
            except Exception as e:
                logger.warning(f"Failed to analyze company position: {str(e)}")
            
            # Analyze competitor positions
            for competitor in competitors[:5]:
                try:
                    position_query = f"{competitor['name']} {selected_dimension[0]} {selected_dimension[1]}"
                    results = exa_searcher.query_exa(position_query, "")
                    
                    dim1_score = 0
                    dim2_score = 0
                    
                    for result in results:
                        snippet = result.get('snippet', '').lower()
                        
                        # Score dimensions (similar to company scoring)
                        if selected_dimension[0] in snippet:
                            context_words = snippet.split()
                            idx = context_words.index(selected_dimension[0])
                            if idx > 0:
                                modifier = context_words[idx - 1]
                                if modifier in ['high', 'higher', 'highest']:
                                    dim1_score += 1
                                elif modifier in ['low', 'lower', 'lowest']:
                                    dim1_score -= 1
                        
                        if selected_dimension[1] in snippet:
                            context_words = snippet.split()
                            idx = context_words.index(selected_dimension[1])
                            if idx > 0:
                                modifier = context_words[idx - 1]
                                if modifier in ['high', 'higher', 'highest']:
                                    dim2_score += 1
                                elif modifier in ['low', 'lower', 'lowest']:
                                    dim2_score -= 1
                    
                    positioning_analysis['positioning_map']['competitor_positions'][competitor['name']] = {
                        selected_dimension[0]: min(max((dim1_score + 3) * 20, 0), 100),
                        selected_dimension[1]: min(max((dim2_score + 3) * 20, 0), 100)
                    }
                except Exception as e:
                    logger.warning(f"Failed to analyze position for {competitor['name']}: {str(e)}")
            
            # Identify differentiation factors
            if company_data.market_position:
                factors = []
                for key, value in company_data.market_position.items():
                    if 'different' in key.lower() or 'unique' in key.lower():
                        factors.append(value)
                positioning_analysis['differentiation_factors'].extend(factors)
            
            # Analyze target audience
            try:
                audience_query = f"{company_data.name} target audience OR customer segment OR ideal customer"
                results = exa_searcher.query_exa(audience_query, "")
                
                for result in results:
                    snippet = result.get('snippet', '')
                    if any(term in snippet.lower() for term in ['target', 'audience', 'customer', 'segment']):
                        positioning_analysis['target_audience'].append({
                            'segment': snippet,
                            'source': result.get('url')
                        })
            except Exception as e:
                logger.warning(f"Failed to analyze target audience: {str(e)}")
            
            # Identify growth opportunities
            try:
                growth_query = f"{company_data.industry} growth opportunities OR market gaps 2024"
                results = exa_searcher.query_exa(growth_query, "")
                
                for result in results:
                    snippet = result.get('snippet', '')
                    if any(term in snippet.lower() for term in ['opportunity', 'growth', 'potential', 'gap']):
                        positioning_analysis['growth_opportunities'].append({
                            'opportunity': snippet,
                            'source': result.get('url')
                        })
            except Exception as e:
                logger.warning(f"Failed to identify growth opportunities: {str(e)}")
            
            return positioning_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market positioning: {str(e)}")
            return positioning_analysis

    def _normalize_metric(self, value: str) -> Optional[float]:
        """Normalize metric values to standard units"""
        try:
            # Remove any currency symbols and spaces
            value = re.sub(r'[$€£¥]|\s', '', value.lower())
            
            # Extract number and unit
            match = re.match(r'(\d+(?:\.\d+)?)(k|m|b|t)?', value)
            if not match:
                return None
            
            number = float(match.group(1))
            unit = match.group(2)
            
            # Convert to standard unit
            multipliers = {
                'k': 1_000,
                'm': 1_000_000,
                'b': 1_000_000_000,
                't': 1_000_000_000_000
            }
            
            if unit:
                number *= multipliers.get(unit, 1)
            
            return number
            
        except Exception:
            return None

    def extract_company_info(self, html_content: str, url: str) -> CompanyData:
        """Extract company information from HTML content with enhanced capabilities"""
        soup = BeautifulSoup(html_content, 'html.parser')
        domain = urlparse(url).netloc
        
        # Enhanced company name extraction
        company_name = None
        name_sources = [
            # Schema.org data
            ('script', {'type': 'application/ld+json'}, lambda x: self._extract_from_json_ld(x, ['name', 'legalName'])),
            # Meta tags
            ('meta', {'property': 'og:site_name'}, lambda x: x.get('content')),
            ('meta', {'property': 'og:title'}, lambda x: x.get('content')),
            ('meta', {'name': 'application-name'}, lambda x: x.get('content')),
            # Common title patterns
            ('title', {}, lambda x: x.string.split('|')[0].strip() if x.string and '|' in x.string else x.string.strip() if x.string else None),
            ('h1', {'class': lambda x: x and any(term in str(x).lower() for term in ['company', 'brand', 'logo', 'site-title'])}, lambda x: x.string),
        ]
        
        for tag, attrs, extractor in name_sources:
            elements = soup.find_all(tag, attrs)
            for element in elements:
                try:
                    extracted = extractor(element)
                    if extracted and len(extracted) > 1:
                        company_name = extracted
                        break
                except Exception:
                    continue
            if company_name:
                break
        
        # Fallback to domain name if no company name found
        if not company_name:
            company_name = domain.split('.')[0].title()
        
        # Enhanced description extraction with multiple sources
        description = None
        desc_sources = [
            # Schema.org data
            ('script', {'type': 'application/ld+json'}, lambda x: self._extract_from_json_ld(x, ['description', 'about'])),
            # Meta tags
            ('meta', {'property': 'og:description'}, lambda x: x.get('content')),
            ('meta', {'name': 'description'}, lambda x: x.get('content')),
            # Common description locations
            ('div', {'class': lambda x: x and any(term in str(x).lower() for term in ['about', 'company', 'description', 'intro'])}, lambda x: x.get_text(strip=True)),
            ('p', {'class': lambda x: x and any(term in str(x).lower() for term in ['about', 'company', 'description', 'intro'])}, lambda x: x.get_text(strip=True)),
        ]
        
        for tag, attrs, extractor in desc_sources:
            elements = soup.find_all(tag, attrs)
            for element in elements:
                try:
                    extracted = extractor(element)
                    if extracted and len(extracted) > 50:  # Ensure meaningful description
                        description = extracted
                        break
                except Exception:
                    continue
            if description:
                break
        
        # Extract company size and employee count
        company_size = None
        employee_patterns = [
            r'(\d+(?:[,\s]\d+)?(?:\+|\s*-\s*\d+)?)\s*(?:employees|team members|people|staff)',
            r'team of (\d+(?:[,\s]\d+)?(?:\+|\s*-\s*\d+)?)',
            r'(\d+(?:[,\s]\d+)?(?:\+|\s*-\s*\d+)?)\s*strong team'
        ]
        
        text_content = soup.get_text()
        for pattern in employee_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                company_size = match.group(1)
                break
        
        # Extract founding year
        founding_year = None
        year_patterns = [
            r'(?:founded|established|started).*?in\s+(\d{4})',
            r'since\s+(\d{4})',
            r'founded\s+(\d{4})'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                founding_year = match.group(1)
                break
        
        # Extract location information
        location = None
        location_patterns = [
            r'(?:headquartered|based|located).*?in\s+([^\.]+)',
            r'(?:main|primary|global)\s+headquarters.*?in\s+([^\.]+)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                break
        
        # Extract funding information
        funding_info = None
        funding_patterns = [
            r'(?:raised|secured|closed)\s+(\$[\d\.]+[MBK]?)\s+(?:in|series)',
            r'(\$[\d\.]+[MBK]?)\s+(?:funding|investment|round)'
        ]
        
        for pattern in funding_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                funding_info = match.group(1)
                break
        
        # Enhanced logo extraction (existing code)
        logo_url = None
        logo_selectors = [
            # Meta tags with high priority
            ('meta', {'property': 'og:logo'}),
            ('meta', {'name': 'og:logo'}),
            ('meta', {'property': 'og:image'}),
            ('meta', {'name': 'og:image'}),
            # Schema.org markup
            ('script', {'type': 'application/ld+json'}),
            # Common logo locations
            ('a', {'class': lambda x: x and any(term in str(x).lower() for term in ['logo', 'brand', 'header-logo', 'site-logo', 'navbar-brand'])}),
            ('div', {'class': lambda x: x and any(term in str(x).lower() for term in ['logo', 'brand', 'header-logo', 'site-logo', 'navbar-brand'])}),
            ('img', {'class': lambda x: x and any(term in str(x).lower() for term in ['logo', 'brand', 'header-logo', 'site-logo', 'navbar-brand'])}),
            # Header logos
            ('header img', {}),
            ('.header img', {}),
            ('#header img', {}),
            ('nav img', {}),
            # Link tags
            ('link', {'rel': 'apple-touch-icon'}),
            ('link', {'rel': 'icon'}),
            ('link', {'rel': 'shortcut icon'}),
            # Common logo IDs
            ('img', {'id': lambda x: x and any(term in str(x).lower() for term in ['logo', 'brand', 'header-logo', 'site-logo'])}),
        ]

        def extract_logo_from_json_ld(script_content):
            try:
                data = json.loads(script_content)
                if isinstance(data, list):
                    data = data[0]
                if isinstance(data, dict):
                    # Check for organization logo
                    logo = data.get('logo') or \
                           data.get('organization', {}).get('logo') or \
                           data.get('publisher', {}).get('logo') or \
                           data.get('image')
                    if isinstance(logo, dict):
                        return logo.get('url')
                    return logo if isinstance(logo, str) else None
            except (json.JSONDecodeError, AttributeError):
                return None
            return None

        def is_valid_logo_url(url_str: str) -> bool:
            """Validate if the URL is likely to be a logo"""
            if not url_str or url_str.startswith('data:'):
                return False
            
            # Check file extension
            valid_extensions = ('.png', '.jpg', '.jpeg', '.svg', '.gif', '.ico', '.webp')
            if any(url_str.lower().endswith(ext) for ext in valid_extensions):
                return True
            
            # Check URL patterns
            logo_indicators = ('logo', 'brand', 'header', 'site-logo', 'company')
            if any(indicator in url_str.lower() for indicator in logo_indicators):
                return True
            
            return False

        # Try each selector in order
        for selector in logo_selectors:
            try:
                if len(selector) == 2:
                    tag, attrs = selector
                    if tag == 'script' and attrs.get('type') == 'application/ld+json':
                        # Handle JSON-LD
                        for script in soup.find_all(tag, attrs):
                            logo_url = extract_logo_from_json_ld(script.string)
                            if logo_url and is_valid_logo_url(logo_url):
                                break
                    else:
                        # Handle regular tags
                        elements = soup.find_all(tag, attrs)
                        for element in elements:
                            if tag == 'img':
                                logo_url = element.get('src')
                            elif tag in ('meta', 'link'):
                                logo_url = element.get('content') or element.get('href')
                            elif tag in ('a', 'div'):
                                img = element.find('img')
                                if img:
                                    logo_url = img.get('src')
                            
                            if logo_url and is_valid_logo_url(logo_url):
                                break
                else:
                    # Handle CSS selector
                    elements = soup.select(selector[0])
                    for element in elements:
                        if element.name == 'img':
                            logo_url = element.get('src')
                            if logo_url and is_valid_logo_url(logo_url):
                                break
                
                if logo_url:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing logo selector {selector}: {str(e)}")
                continue
                
        # Normalize logo URL
        if logo_url:
            try:
                if logo_url.startswith('data:'):
                    logo_url = None
                else:
                    logo_url = urljoin(url, logo_url)
                    # Verify URL format
                    parsed = urlparse(logo_url)
                    if not all([parsed.scheme, parsed.netloc]):
                        logo_url = None
            except Exception as e:
                logger.warning(f"Error normalizing logo URL: {str(e)}")
                logo_url = None

        if not logo_url:
            logger.warning(f"No logo found for {url}")

        # Enhanced social links extraction
        social_links = []
        social_patterns = {
            'facebook.com': r'facebook\.com/[^/\s"\']+',
            'twitter.com': r'twitter\.com/[^/\s"\']+',
            'linkedin.com': r'linkedin\.com/(?:company|in|school)/[^/\s"\']+',
            'instagram.com': r'instagram\.com/[^/\s"\']+',
            'youtube.com': r'youtube\.com/(?:channel|user|c)/[^/\s"\']+',
            'github.com': r'github\.com/[^/\s"\']+',
            'medium.com': r'medium\.com/(?:@)?[^/\s"\']+',
            'crunchbase.com': r'crunchbase\.com/organization/[^/\s"\']+',
        }
        
        for link in soup.find_all(['a', 'link'], href=True):
            href = link['href']
            for domain, pattern in social_patterns.items():
                if domain in href.lower():
                    match = re.search(pattern, href, re.I)
                    if match:
                        full_url = urljoin(url, href)
                        if full_url not in social_links:
                            social_links.append(full_url)
        
        # Extract media assets (existing code)
        media_assets = []
        seen_urls = set()
        
        def is_valid_image_url(img_url: str) -> bool:
            """Validate image URL and check if it's worth including"""
            if not img_url or img_url.startswith('data:'):
                return False
            
            # Check file extension
            valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp')
            if not any(img_url.lower().endswith(ext) for ext in valid_extensions):
                return False
            
            # Skip common irrelevant images
            skip_patterns = ('spacer', 'pixel', 'tracking', 'analytics', 'advertisement', 'ad/')
            if any(pattern in img_url.lower() for pattern in skip_patterns):
                return False
            
            return True

        def is_relevant_image(img_tag) -> bool:
            """Check if the image is relevant based on various factors"""
            # Skip if no src
            if not img_tag.get('src'):
                return False
            
            # Check dimensions if available
            width = img_tag.get('width')
            height = img_tag.get('height')
            if width and height:
                try:
                    w, h = int(width), int(height)
                    if w < 100 or h < 100:  # Skip tiny images
                        return False
                except ValueError:
                    pass
            
            # Check alt text and surrounding context
            alt_text = img_tag.get('alt', '').lower()
            parent_text = ' '.join(p.get_text(' ', strip=True).lower() for p in img_tag.parents if p.name in ['div', 'section'])
            
            relevant_terms = [
                'product', 'company', 'logo', 'team', 'office', 'building',
                'service', 'solution', 'feature', 'banner', 'hero', 'about',
                'portfolio', 'project', 'work'
            ]
            
            return any(term in alt_text or term in parent_text for term in relevant_terms)

        # Process images with improved relevance checking
        for img in soup.find_all('img', src=True):
            try:
                src = img['src']
                if not is_valid_image_url(src):
                    continue
                
                # Normalize URL
                img_url = urljoin(url, src)
                if img_url in seen_urls:
                    continue
                
                if not is_relevant_image(img):
                    continue
                
                seen_urls.add(img_url)
                
                # Extract metadata
                metadata = {
                    'width': img.get('width'),
                    'height': img.get('height'),
                    'loading': img.get('loading', 'eager'),
                    'title': img.get('title', ''),
                    'class': img.get('class', []),
                    'id': img.get('id', ''),
                }
                
                media_assets.append(MediaAsset(
                    type='image',
                    url=img_url,
                    alt=img.get('alt', ''),
                    metadata=metadata
                ))
            except Exception as e:
                logger.warning(f"Error processing image: {str(e)}")
                continue
        
        # Extract technologies used
        technologies = self.extract_technologies(soup)
        
        # Extract key metrics
        key_metrics = {
            'company_size': company_size,
            'founding_year': founding_year,
            'location': location,
            'funding': funding_info
        }
        
        # Create CompanyData instance with enhanced information
        company_data = CompanyData(
            name=company_name,
            domain=domain,
            description=description,
            logo_url=logo_url,
            social_links=social_links,
            media_assets=media_assets,
            product_lines=self.extract_product_lines(soup),
            competitors=[],  # Will be populated by find_competitors
            industry=None,  # Will be determined by analyze_company_data
            key_metrics=key_metrics,
            technologies=technologies,
            executives=self.extract_executives(soup),
            funding_rounds=self.extract_funding_info(soup),
            documents=self.extract_documents(soup, url),
            videos=self.extract_videos(soup, url),
            achievements=self.extract_achievements(soup),
            financial_metrics=self.extract_financial_metrics(soup),
            market_position=self.extract_market_position(soup),
            growth_indicators=self.extract_growth_indicators(soup),
            timeline_events=[],  # Will be populated by extract_timeline_events
            benchmarking_data={},  # Will be populated by analyze_benchmarking_data
            positioning_analysis={},  # Will be populated by analyze_market_positioning
            data_sources=None  # Will be populated by ScrapingAnt
        )
        
        return company_data

    def _extract_from_json_ld(self, element, keys: List[str]) -> Optional[str]:
        """Helper method to extract data from JSON-LD"""
        try:
            data = json.loads(element.string)
            if isinstance(data, list):
                data = data[0]
            
            if isinstance(data, dict):
                for key in keys:
                    value = data.get(key)
                    if isinstance(value, dict):
                        return value.get('name') or value.get('text')
                    elif isinstance(value, str):
                        return value
            return None
        except (json.JSONDecodeError, AttributeError):
            return None

    def extract_videos(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract video content from the website"""
        videos = []
        
        # Look for video elements and iframes
        video_elements = (
            soup.find_all('video', src=True) +
            soup.find_all('iframe', src=True) +
            soup.find_all('a', href=True, class_=lambda x: x and 'video' in x.lower()) +
            soup.find_all('div', class_=lambda x: x and any(term in str(x).lower() for term in ['video', 'player'])) +
            soup.find_all('source', type=lambda x: x and 'video' in x.lower())
        )
        
        # Video platform patterns
        video_platforms = {
            'youtube': [r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)',
                       r'youtube\.com/embed/([a-zA-Z0-9_-]+)'],
            'vimeo': [r'vimeo\.com/(\d+)',
                     r'player\.vimeo\.com/video/(\d+)'],
            'wistia': [r'fast\.wistia\.net/embed/iframe/([a-zA-Z0-9]+)',
                      r'wistia\.com/medias/([a-zA-Z0-9]+)'],
            'loom': [r'loom\.com/share/([a-zA-Z0-9]+)',
                    r'loom\.com/embed/([a-zA-Z0-9]+)'],
            'vidyard': [r'vidyard\.com/watch/([a-zA-Z0-9-]+)',
                       r'play\.vidyard\.com/([a-zA-Z0-9-]+)']
        }
        
        for element in video_elements:
            src = element.get('src') or element.get('href') or element.get('data-url')
            if not src:
                # Check for nested video sources
                sources = element.find_all('source', src=True)
                if sources:
                    src = sources[0].get('src')
            
            if src:
                video_info = {
                    'type': 'unknown',
                    'url': urljoin(base_url, src),
                    'title': '',
                    'description': '',
                    'thumbnail': '',
                    'duration': '',
                    'platform': ''
                }
                
                # Get title and description
                video_info['title'] = (
                    element.get('title') or 
                    element.get('alt') or 
                    element.get('aria-label') or
                    element.get('data-title') or
                    ''
                )
                
                video_info['description'] = (
                    element.get('description') or
                    element.get('data-description') or
                    element.get('aria-description') or
                    ''
                )
                
                # Get thumbnail if available
                thumbnail = (
                    element.get('poster') or
                    element.get('data-thumbnail') or
                    element.get('data-thumb')
                )
                if thumbnail:
                    video_info['thumbnail'] = urljoin(base_url, thumbnail)
                
                # Get duration if available
                duration = element.get('duration') or element.get('data-duration')
                if duration:
                    video_info['duration'] = duration
                
                # Identify video platform and format URL
                platform_found = False
                for platform, patterns in video_platforms.items():
                    for pattern in patterns:
                        match = re.search(pattern, src)
                        if match:
                            video_id = match.group(1)
                            video_info['platform'] = platform
                            video_info['type'] = platform
                            
                            # Format URL based on platform
                            if platform == 'youtube':
                                video_info['url'] = f'https://www.youtube.com/embed/{video_id}'
                                if not video_info['thumbnail']:
                                    video_info['thumbnail'] = f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg'
                            elif platform == 'vimeo':
                                video_info['url'] = f'https://player.vimeo.com/video/{video_id}'
                            elif platform == 'wistia':
                                video_info['url'] = f'https://fast.wistia.net/embed/iframe/{video_id}'
                            elif platform == 'loom':
                                video_info['url'] = f'https://www.loom.com/embed/{video_id}'
                            elif platform == 'vidyard':
                                video_info['url'] = f'https://play.vidyard.com/{video_id}'
                            
                            platform_found = True
                            break
                    if platform_found:
                        break
                
                # Handle direct video files
                if not platform_found:
                    if src.endswith(('.mp4', '.webm', '.ogg', '.mov', '.m4v')):
                        video_info['type'] = 'direct'
                        video_info['platform'] = 'html5'
                    elif 'brightcove' in src.lower():
                        video_info['type'] = 'brightcove'
                        video_info['platform'] = 'brightcove'
                    elif 'kaltura' in src.lower():
                        video_info['type'] = 'kaltura'
                        video_info['platform'] = 'kaltura'
                
                # Only add videos we can identify
                if video_info['type'] != 'unknown':
                    videos.append(video_info)
        
        return videos

    def extract_documents(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract relevant documents from the website"""
        documents = []
        
        # Look for document links
        doc_patterns = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']
        doc_keywords = ['annual report', 'investor', 'presentation', 'whitepaper', 'case study']
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True).lower()
            
            if any(pattern in href.lower() for pattern in doc_patterns) or any(keyword in text for keyword in doc_keywords):
                documents.append({
                    'type': 'document',
                    'url': urljoin(base_url, href),
                    'title': link.get_text(strip=True) or href.split('/')[-1],
                    'format': href.split('.')[-1] if '.' in href else 'unknown'
                })
        
        return documents

    def extract_executives(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract executive team information"""
        executives = []
        
        # Common patterns for executive sections
        team_sections = soup.find_all(['section', 'div'], class_=lambda x: x and any(
            term in x.lower() for term in ['team', 'leadership', 'management', 'executive', 'board']
        ))
        
        for section in team_sections:
            # Look for executive profiles
            profiles = section.find_all(['div', 'article'], class_=lambda x: x and any(
                term in x.lower() for term in ['profile', 'member', 'card', 'bio']
            ))
            
            for profile in profiles:
                name = profile.find(['h2', 'h3', 'h4', 'strong'])
                title = profile.find(['p', 'div'], class_=lambda x: x and any(
                    term in x.lower() for term in ['title', 'position', 'role']
                ))
                bio = profile.find(['p', 'div'], class_=lambda x: x and any(
                    term in x.lower() for term in ['bio', 'description', 'about']
                ))
                
                if name:
                    executives.append({
                        'name': name.get_text(strip=True),
                        'title': title.get_text(strip=True) if title else '',
                        'bio': bio.get_text(strip=True) if bio else '',
                        'linkedin': next((
                            link['href'] for link in profile.find_all('a', href=True)
                            if 'linkedin.com' in link['href'].lower()
                        ), '')
                    })
        
        return executives

    def extract_funding_info(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract funding information"""
        funding_rounds = []
        
        # Look for funding-related content
        funding_sections = soup.find_all(['section', 'div'], class_=lambda x: x and any(
            term in x.lower() for term in ['funding', 'investment', 'investor', 'round']
        ))
        
        for section in funding_sections:
            # Extract individual funding rounds
            rounds = section.find_all(['div', 'article'], class_=lambda x: x and any(
                term in x.lower() for term in ['round', 'series', 'investment']
            ))
            
            for round_info in rounds:
                amount = round_info.find(text=re.compile(r'[\$£€]?\d+(?:\.\d+)?[MBK]?'))
                date = round_info.find(text=re.compile(r'\d{4}'))
                investors = round_info.find_all(['a', 'span'], class_=lambda x: x and 'investor' in str(x).lower())
                
                funding_rounds.append({
                    'round_type': round_info.get_text(strip=True)[:50],
                    'amount': amount.strip() if amount else '',
                    'date': date.strip() if date else '',
                    'investors': [inv.get_text(strip=True) for inv in investors]
                })
        
        return funding_rounds

    def extract_achievements(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract company achievements and milestones"""
        achievements = []
        
        # Look for achievement sections
        achievement_sections = soup.find_all(['section', 'div'], class_=lambda x: x and any(
            term in x.lower() for term in ['achievement', 'milestone', 'award', 'recognition']
        ))
        
        for section in achievement_sections:
            items = section.find_all(['li', 'div', 'article'], class_=lambda x: x and any(
                term in x.lower() for term in ['item', 'achievement', 'milestone']
            ))
            
            for item in items:
                title = item.find(['h3', 'h4', 'strong'])
                date = item.find(text=re.compile(r'\d{4}'))
                description = item.find(['p', 'div'], class_=lambda x: x and 'description' in str(x).lower())
                
                achievements.append({
                    'title': title.get_text(strip=True) if title else '',
                    'date': date.strip() if date else '',
                    'description': description.get_text(strip=True) if description else ''
                })
        
        return achievements

    def extract_financial_metrics(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract financial metrics and KPIs"""
        metrics = {}
        
        # Look for financial information
        financial_sections = soup.find_all(['section', 'div'], class_=lambda x: x and any(
            term in x.lower() for term in ['financial', 'metrics', 'kpi', 'performance']
        ))
        
        for section in financial_sections:
            # Look for common financial metrics
            metric_patterns = [
                r'revenue:?\s*[\$£€]?\d+(?:\.\d+)?[MBK]?',
                r'growth:?\s*\d+(?:\.\d+)?%',
                r'margin:?\s*\d+(?:\.\d+)?%',
                r'ebitda:?\s*[\$£€]?\d+(?:\.\d+)?[MBK]?'
            ]
            
            for pattern in metric_patterns:
                matches = section.find_all(text=re.compile(pattern, re.IGNORECASE))
                for match in matches:
                    key = match.split(':')[0].strip()
                    value = match.split(':')[1].strip()
                    metrics[key] = value
        
        return metrics

    def extract_market_position(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract market position and competitive advantage information"""
        position = {}
        
        # Look for market position indicators
        market_sections = soup.find_all(['section', 'div'], class_=lambda x: x and any(
            term in x.lower() for term in ['market', 'position', 'advantage', 'leadership']
        ))
        
        for section in market_sections:
            # Extract market share
            market_share = section.find(text=re.compile(r'market share:?\s*\d+(?:\.\d+)?%', re.IGNORECASE))
            if market_share:
                position['market_share'] = market_share.strip()
            
            # Extract market size
            market_size = section.find(text=re.compile(r'market size:?\s*[\$£€]?\d+(?:\.\d+)?[MBK]?', re.IGNORECASE))
            if market_size:
                position['market_size'] = market_size.strip()
            
            # Extract competitive advantages
            advantages = section.find_all(['li', 'p'], class_=lambda x: x and 'advantage' in str(x).lower())
            if advantages:
                position['advantages'] = [adv.get_text(strip=True) for adv in advantages]
        
        return position

    def extract_growth_indicators(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract growth indicators and trends"""
        indicators = []
        
        # Look for growth-related content
        growth_sections = soup.find_all(['section', 'div'], class_=lambda x: x and any(
            term in x.lower() for term in ['growth', 'expansion', 'trend']
        ))
        
        for section in growth_sections:
            # Extract growth metrics
            metrics = section.find_all(['p', 'div'], class_=lambda x: x and any(
                term in x.lower() for term in ['metric', 'stat', 'indicator']
            ))
            
            for metric in metrics:
                value = metric.find(text=re.compile(r'\d+(?:\.\d+)?[%MBK]?'))
                description = metric.get_text(strip=True)
                
                if value:
                    indicators.append({
                        'metric': description.replace(value, '').strip(),
                        'value': value.strip(),
                        'period': next((
                            period.strip() for period in metric.strings
                            if any(term in period.lower() for term in ['year', 'quarter', 'month'])
                        ), '')
                    })
        
        return indicators

    # --- New wrapper method to address missing extract_technologies ---
    def extract_technologies(self, soup: BeautifulSoup) -> List[str]:
        """Wrapper method that calls the standalone extract_technologies function"""
        return extract_technologies(soup)

def save_to_history(company_data: CompanyData):
    """Save company data to history CSV file"""
    # Define the columns we want to save
    columns = [
        'scrape_date', 'name', 'domain', 'description', 'logo_url',
        'social_links', 'media_assets', 'product_lines', 'competitors',
        'industry', 'key_metrics', 'technologies', 'executives', 'funding_rounds',
        'documents', 'videos', 'achievements', 'financial_metrics', 'market_position',
        'growth_indicators', 'timeline_events', 'benchmarking_data', 'positioning_analysis'
    ]
    
    # Ensure data is properly formatted
    data = {
        'scrape_date': company_data.scrape_date.strftime('%Y-%m-%d %H:%M:%S'),
        'name': str(company_data.name) if company_data.name else '',
        'domain': str(company_data.domain) if company_data.domain else '',
        'description': str(company_data.description) if company_data.description else '',
        'logo_url': str(company_data.logo_url) if company_data.logo_url else '',
        'social_links': str(company_data.social_links) if company_data.social_links else '[]',
        'media_assets': str([{
            'type': str(asset.type),
            'url': str(asset.url),
            'alt': str(asset.alt),
            'metadata': asset.metadata
        } for asset in company_data.media_assets]) if company_data.media_assets else '[]',
        'product_lines': str(company_data.product_lines) if company_data.product_lines else '[]',
        'competitors': str(company_data.competitors) if company_data.competitors else '[]',
        'industry': str(company_data.industry) if company_data.industry else '',
        'key_metrics': str(company_data.key_metrics) if company_data.key_metrics else '{}',
        'technologies': str(company_data.technologies) if company_data.technologies else '[]',
        'executives': str(company_data.executives) if company_data.executives else '[]',
        'funding_rounds': str(company_data.funding_rounds) if company_data.funding_rounds else '[]',
        'documents': str(company_data.documents) if company_data.documents else '[]',
        'videos': str(company_data.videos) if company_data.videos else '[]',
        'achievements': str(company_data.achievements) if company_data.achievements else '[]',
        'financial_metrics': str(company_data.financial_metrics) if company_data.financial_metrics else '{}',
        'market_position': str(company_data.market_position) if company_data.market_position else '{}',
        'growth_indicators': str(company_data.growth_indicators) if company_data.growth_indicators else '[]',
        'timeline_events': str(company_data.timeline_events) if company_data.timeline_events else '[]',
        'benchmarking_data': str(company_data.benchmarking_data) if company_data.benchmarking_data else '{}',
        'positioning_analysis': str(company_data.positioning_analysis) if company_data.positioning_analysis else '{}'
    }
    
    try:
        if os.path.exists('company_data_history.csv'):
            # Read existing data with proper timestamp parsing
            existing_df = pd.read_csv('company_data_history.csv')
            existing_df['scrape_date'] = pd.to_datetime(existing_df['scrape_date'])
            
            # Create new DataFrame with current data
            new_df = pd.DataFrame([data], columns=columns)
            new_df['scrape_date'] = pd.to_datetime(new_df['scrape_date'])
            
            # Ensure all columns exist in both DataFrames
            for col in columns:
                if col not in existing_df.columns:
                    existing_df[col] = ''
            
            # Combine DataFrames
            df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates based on domain and scrape_date
            df = df.sort_values('scrape_date', ascending=False)
            df = df.drop_duplicates(subset=['domain'], keep='first')
        else:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame([data], columns=columns)
            df['scrape_date'] = pd.to_datetime(df['scrape_date'])
        
        # Save to CSV with consistent columns
        df.to_csv('company_data_history.csv', index=False)
        logger.info(f"Successfully saved data for {company_data.domain} to history")
        
    except Exception as e:
        logger.error(f"Error saving to history file: {str(e)}")
        # Create new file if there was an error
        new_df = pd.DataFrame([data], columns=columns)
        new_df.to_csv('company_data_history.csv', index=False)
        logger.info("Created new history file with current data")

def analyze_company_data(company_url: str) -> Dict:
    results = {"success": False, "company_data": None, "metrics": {}, "error": None}
    logger.info(f"Analyzing company data for: {company_url}")
    
    try:
        validate_api_keys()
        scraper = ScrapingAntScraper(SCRAPINGANT_API_KEY)
        html_content = scraper.scrape_page(company_url)
        
        if not html_content:
            results["error"] = "Failed to scrape website content"
            return results
        
        soup = BeautifulSoup(html_content, 'html.parser')
        scraped_data = scraper.extract_company_info(html_content, company_url)
        
        if not scraped_data:
            results["error"] = "Failed to extract company information"
            return results
            
        # Initialize default values for required fields
        company_name = scraped_data.get('name') or urlparse(company_url).netloc
        
        exa_searcher = ExaSearcher(EXA_API_KEY)
        search_results = exa_searcher.query_exa(company_name, company_url)
        exa_data = exa_searcher.extract_company_info(search_results) if search_results else None
        
        technologies = extract_technologies(soup)
        key_metrics = extract_key_metrics(soup, scraped_data.get('description', ''))
        
        industry = None
        if exa_data and getattr(exa_data, 'description', None):
            industry_keywords = ["technology", "software", "healthcare", "retail", "finance", "manufacturing", "automotive", "energy", "telecommunications", "education", "real estate", "construction", "agriculture"]
            description_lower = exa_data.description.lower()
            for keyword in industry_keywords:
                if keyword in description_lower:
                    industry = keyword.title()
                    break
        
        # Initialize competitors before using find_competitors
        competitors = find_competitors(company_name, industry, exa_searcher) if industry else []
        
        if exa_data:
            if getattr(exa_data, 'description', None) and (not scraped_data.get('description') or len(exa_data.description) > len(scraped_data.get('description', ''))):
                scraped_data['description'] = exa_data.description
            if not scraped_data.get('name') or scraped_data.get('name') == scraped_data.get('domain', '').split('.')[0]:
                scraped_data['name'] = getattr(exa_data, 'name', company_name)
        
        # Update scraped data with new information
        scraped_data.update({
            'technologies': technologies,
            'key_metrics': key_metrics,
            'competitors': competitors,
            'industry': industry
        })
        
        metrics = {
            "social_presence_score": len(scraped_data.get('social_links', [])) / 4.0 * 100,
            "media_richness_score": min(len(scraped_data.get('media_assets', [])) / 10 * 100, 100),
            "information_completeness": sum(1 for v in [
                scraped_data.get('name'), 
                scraped_data.get('description'), 
                scraped_data.get('logo_url'), 
                scraped_data.get('social_links'), 
                scraped_data.get('technologies'), 
                scraped_data.get('competitors')
            ] if v) / 6.0 * 100
        }
        
        # Ensure all required fields exist before saving
        company_dict = {
            "name": scraped_data.get('name', company_name),
            "domain": scraped_data.get('domain', urlparse(company_url).netloc),
            "description": scraped_data.get('description', ''),
            "logo_url": scraped_data.get('logo_url', ''),
            "social_links": scraped_data.get('social_links', []),
            "media_assets": [
                MediaAsset(
                    type=asset.get('type', ''),
                    url=asset.get('url', ''),
                    alt=asset.get('alt', ''),
                    metadata=asset.get('metadata', {})
                ) if isinstance(asset, dict) else asset
                for asset in scraped_data.get('media_assets', [])
            ],
            "product_lines": scraped_data.get('product_lines', []),
            "competitors": scraped_data.get('competitors', []),
            "industry": scraped_data.get('industry', ''),
            "key_metrics": scraped_data.get('key_metrics', {}),
            "technologies": scraped_data.get('technologies', []),
            "scrape_date": datetime.now().isoformat()
        }
        
        results.update({
            "success": True,
            "company_data": company_dict,
            "metrics": metrics
        })
    except ValueError as e:
        logger.error(f"Validation Error: {str(e)}")
        results["error"] = str(e)
    except Exception as e:
        logger.error(f"Analysis Exception: {str(e)}")
        results["error"] = str(e)
    
    return results

def get_api_usage_stats() -> Dict:
    """Get API usage statistics"""
    try:
        df = pd.read_csv("api_usage.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate statistics
        stats = {
            'total_requests': len(df),
            'total_credits_used': df['credits_used'].sum(),
            'total_cost': df['cost'].sum(),
            'average_response_time': df['response_time'].mean(),
            'success_rate': (df['status'] == '200').mean() * 100,
            'usage_by_day': df.groupby(df['timestamp'].dt.date).size().to_dict(),
            'cost_by_day': df.groupby(df['timestamp'].dt.date)['cost'].sum().to_dict()
        }
        
        return {
            'success': True,
            'stats': stats
        }
    except Exception as e:
        logger.error(f"Error getting API usage stats: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def compare_approaches(company_url: str, company_name: str):
    """Compare different approaches for gathering company data"""
    
    results = {
        "scrapingant": {"success": False, "data": None, "error": None},
        "exa": {"success": False, "data": None, "error": None}
    }
    
    # 1. ScrapingAnt Approach
    logger.info("Testing ScrapingAnt approach...")
    try:
        scraper = ScrapingAntScraper(SCRAPINGANT_API_KEY)
        html_content = scraper.scrape_page(company_url)
        if html_content:
            results["scrapingant"] = {
                "success": True,
                "data": {
                    "raw_html": len(html_content),
                    "can_access_dynamic_content": True,
                    "supports_javascript": True,
                    "rate_limits": "100,000 credits for $19/month"
                }
            }
    except Exception as e:
        results["scrapingant"]["error"] = str(e)

    # 2. Exa.ai Approach
    logger.info("Testing Exa.ai approach...")
    try:
        searcher = ExaSearcher(EXA_API_KEY)
        search_results = searcher.query_exa(company_name, company_url)
        if search_results:
            results["exa"] = {
                "success": True,
                "data": {
                    "num_results": len(search_results),
                    "content_types": "text, structured data",
                    "search_capabilities": "advanced semantic search",
                    "rate_limits": "based on API plan"
                }
            }
    except Exception as e:
        results["exa"]["error"] = str(e)

    # Print comparison results
    print("\n=== Approach Comparison Results ===\n")
    
    print("ScrapingAnt Advantages:")
    print("✓ Direct HTML access")
    print("✓ JavaScript rendering support")
    print("✓ Good for media asset extraction")
    print("✓ Can handle dynamic content")
    print("\nScrapingAnt Limitations:")
    print("× Higher resource usage")
    print("× Needs additional parsing logic")
    print("× May require multiple requests")
    
    print("\nExa.ai Advantages:")
    print("✓ Semantic search capabilities")
    print("✓ Pre-processed structured data")
    print("✓ Good for company information")
    print("✓ Single API call for multiple sources")
    print("\nExa.ai Limitations:")
    print("× Limited direct media access")
    print("× May miss real-time updates")
    print("× Dependent on indexed content")
    
    print("\n=== Recommended Hybrid Approach ===\n")
    print("1. Use Exa.ai for:")
    print("   - Company overview information")
    print("   - Industry analysis")
    print("   - News and updates")
    
    print("\n2. Use ScrapingAnt for:")
    print("   - Media asset extraction")
    print("   - Real-time content updates")
    print("   - Dynamic content access")
    
    return results

def extract_technologies(soup: BeautifulSoup) -> List[str]:
    """Extract technology stack information from website"""
    technologies = set()
    
    # Common technology indicators
    tech_patterns = [
        "react", "angular", "vue", "node", "python", "java", "aws", "azure",
        "google cloud", "kubernetes", "docker", "mongodb", "postgresql",
        "mysql", "redis", "elasticsearch", "kafka", "rabbitmq"
    ]
    
    # Check meta tags and scripts
    for meta in soup.find_all("meta"):
        content = meta.get("content", "").lower()
        for tech in tech_patterns:
            if tech in content:
                technologies.add(tech)
    
    for script in soup.find_all("script"):
        src = script.get("src", "").lower()
        for tech in tech_patterns:
            if tech in src:
                technologies.add(tech)
    
    # Check for common technology footprints
    html_content = str(soup).lower()
    for tech in tech_patterns:
        if tech in html_content:
            technologies.add(tech)
    
    return list(technologies)

def extract_key_metrics(soup: BeautifulSoup, description: str) -> Dict[str, str]:
    """Extract key business metrics from website and description"""
    metrics = {}
    
    # Look for common metric patterns
    metric_patterns = [
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|billion|trillion)",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:employees|customers|users)",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:countries|markets|locations)",
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:%|percent)"
    ]
    
    # Search in text content
    text_content = soup.get_text() + " " + (description or "")
    
    for pattern in metric_patterns:
        matches = re.finditer(pattern, text_content, re.IGNORECASE)
        for match in matches:
            context = text_content[max(0, match.start() - 50):min(len(text_content), match.end() + 50)]
            key = context.strip()
            value = match.group(0)
            metrics[key] = value
    
    return metrics

def find_competitors(company_name: str, industry: str, exa_searcher: ExaSearcher) -> List[Dict[str, str]]:
    """Find and analyze competitors with enhanced functionality"""
    try:
        # Build a more comprehensive search query
        search_query = f"{company_name} competitors OR alternatives OR similar companies in {industry}"
        results = exa_searcher.query_exa(search_query, "")
        
        competitors = []
        seen_companies = set()
        
        for result in results:
            # Extract competitor information from search results
            snippet = result.get('snippet', '').lower()
            title = result.get('title', '').lower()
            url = result.get('url', '')
            
            # Skip if the result is about the company itself
            if company_name.lower() in url.lower():
                continue
            
            # Look for competitor mentions in specific patterns
            patterns = [
                r'competitors? (?:include|are|:)\s*([^\.]+)',
                r'alternatives? to [^:]+:\s*([^\.]+)',
                r'similar (?:to|companies|products):\s*([^\.]+)',
                r'compared to ([^\.]+)',
                r'vs\.? ([^\.]+)',
                r'competition (?:includes|from)\s*([^\.]+)'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, snippet + ' ' + title)
                for match in matches:
                    companies = match.group(1).split(',')
                    for company in companies:
                        # Clean up company name
                        company = re.sub(r'(?:and|&)\s+', '', company)
                        company = company.strip('., \t\n\r')
                        
                        # Skip if too short or already seen
                        if len(company) < 3 or company.lower() in seen_companies:
                            continue
                        
                        # Create competitor info
                        competitor_info = {
                            'name': company.title(),
                            'source_url': url,
                            'relevance_score': 0,  # Will be calculated
                            'market_overlap': [],  # Will be populated
                            'description': '',
                            'comparison_points': []
                        }
                        
                        # Extract comparison points
                        comparison_patterns = [
                            r'better (?:in|at|for) ([^\.]+)',
                            r'specializes? in ([^\.]+)',
                            r'known for ([^\.]+)',
                            r'offers? ([^\.]+)',
                            r'provides? ([^\.]+)'
                        ]
                        
                        for comp_pattern in comparison_patterns:
                            comp_matches = re.finditer(comp_pattern, snippet)
                            for comp_match in comp_matches:
                                point = comp_match.group(1).strip()
                                if point and point not in competitor_info['comparison_points']:
                                    competitor_info['comparison_points'].append(point)
                        
                        # Calculate relevance score based on various factors
                        relevance_factors = {
                            'mention_count': snippet.count(company.lower()) + title.count(company.lower()),
                            'has_comparison': len(competitor_info['comparison_points']) > 0,
                            'source_quality': 'review' in url or 'comparison' in url,
                            'recent_mention': 'new' in snippet or '2024' in snippet or '2023' in snippet
                        }
                        
                        competitor_info['relevance_score'] = (
                            relevance_factors['mention_count'] * 20 +
                            relevance_factors['has_comparison'] * 30 +
                            relevance_factors['source_quality'] * 25 +
                            relevance_factors['recent_mention'] * 25
                        )
                        
                        # Try to find market overlap
                        market_terms = [
                            'market', 'industry', 'sector', 'niche', 'segment',
                            'customer', 'user', 'client', 'audience'
                        ]
                        for term in market_terms:
                            pattern = fr"{term}[^.]+?{company}[^.]+?\."
                            overlap_match = re.search(pattern, snippet, re.IGNORECASE)
                            if overlap_match:
                                overlap = overlap_match.group(0).strip()
                                if overlap not in competitor_info['market_overlap']:
                                    competitor_info['market_overlap'].append(overlap)
                        
                        # Add to competitors list if we have meaningful information
                        if competitor_info['relevance_score'] > 0:
                            competitors.append(competitor_info)
                            seen_companies.add(company.lower())
        
        # Sort competitors by relevance score
        competitors.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Try to get additional information for top competitors
        for competitor in competitors[:5]:
            try:
                # Search for specific competitor information
                comp_query = f"{competitor['name']} company overview description"
                comp_results = exa_searcher.query_exa(comp_query, "")
                
                if comp_results:
                    # Get the most relevant description
                    competitor['description'] = comp_results[0].get('snippet', '')
            except Exception as e:
                logger.warning(f"Failed to get additional info for {competitor['name']}: {str(e)}")
        
        return competitors[:10]  # Return top 10 competitors
        
    except Exception as e:
        logger.error(f"Error finding competitors: {str(e)}")
        return []

def batch_analyze_companies(urls: List[str], max_retries: int = 3, delay_between_requests: int = 2) -> Dict[str, Dict]:
    """Analyze multiple companies in batch with retry logic and rate limiting"""
    results = {}
    total_urls = len(urls)
    
    logger.info(f"Starting batch analysis of {total_urls} companies")
    
    for index, url in enumerate(urls, 1):
        try:
            logger.info(f"Processing {index}/{total_urls}: {url}")
            
            # Check if we already have recent data (within last 7 days)
            if os.path.exists('company_data_history.csv'):
                df = pd.read_csv('company_data_history.csv', parse_dates=['scrape_date'])
                domain = urlparse(url).netloc
                recent_data = df[
                    (df['domain'] == domain) & 
                    (pd.to_datetime(df['scrape_date']) > (datetime.now() - timedelta(days=7)))
                ]
                
                if not recent_data.empty:
                    logger.info(f"Using cached data for {domain} (less than 7 days old)")
                    continue
        except Exception as e:
            logger.error(f"Error checking cached data for {url}: {str(e)}")
            continue
        
        # Analyze company with retries
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                result = analyze_company_data(url)
                if result["success"]:
                    results[url] = result
                    success = True
                else:
                    retry_count += 1
                    logger.warning(f"Attempt {retry_count}/{max_retries} failed for {url}: {result['error']}")
                    time.sleep(delay_between_requests * retry_count)  # Exponential backoff
            except Exception as e:
                retry_count += 1
                logger.error(f"Error processing {url}: {str(e)}")
                time.sleep(delay_between_requests * retry_count)
        
        if not success:
            results[url] = {
                "success": False,
                "error": f"Failed after {max_retries} attempts"
            }
        
        # Add delay between requests to avoid rate limiting
        time.sleep(delay_between_requests)
    
    return results

if __name__ == "__main__":
    # Example usage
    company_url = "https://example.com"
    results = analyze_company_data(company_url)
    
    if results["success"]:
        print("\n=== Company Analysis Results ===\n")
        print(f"Company Name: {results['company_data']['name']}")
        print(f"Domain: {results['company_data']['domain']}")
        print(f"Description: {results['company_data']['description']}")
        print(f"\nMetrics:")
        for metric, value in results['metrics'].items():
            print(f"{metric.replace('_', ' ').title()}: {value:.1f}%")
        print(f"\nSocial Links Found: {len(results['company_data']['social_links'])}")
        print(f"Media Assets Found: {len(results['company_data']['media_assets'])}")
    else:
        print(f"Analysis failed: {results['error']}") 
