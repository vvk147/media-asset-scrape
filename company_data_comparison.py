import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import json
from urllib.parse import urlparse, urljoin
import logging
import csv
from pathlib import Path
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_key(key_name: str) -> Optional[str]:
    """Get API key from environment or return None."""
    return os.getenv(key_name)

# API Keys will be set dynamically when needed
SCRAPINGANT_API_KEY: Optional[str] = None
EXA_API_KEY: Optional[str] = None

def set_api_keys(scrapingant_key: Optional[str] = None, exa_key: Optional[str] = None):
    """Set API keys globally, with fallback to environment variables."""
    global SCRAPINGANT_API_KEY, EXA_API_KEY
    
    # If running locally (not in Streamlit Cloud), use environment variables
    is_streamlit_cloud = os.getenv("STREAMLIT_DEPLOYMENT") == "true"
    
    if not is_streamlit_cloud:
        # In local environment, prefer environment variables
        SCRAPINGANT_API_KEY = os.getenv("SCRAPINGANT_API_KEY")
        EXA_API_KEY = os.getenv("EXA_API_KEY")
        logger.info("Using API keys from environment variables")
    else:
        # In Streamlit Cloud, use provided keys
        SCRAPINGANT_API_KEY = scrapingant_key
        EXA_API_KEY = exa_key
        logger.info("Using API keys from Streamlit configuration")

def validate_api_keys():
    """Validate that required API keys are set."""
    global SCRAPINGANT_API_KEY, EXA_API_KEY
    
    # If keys are not set, try to load from environment
    if not SCRAPINGANT_API_KEY or not EXA_API_KEY:
        set_api_keys()
    
    if not SCRAPINGANT_API_KEY or not EXA_API_KEY:
        is_streamlit_cloud = os.getenv("STREAMLIT_DEPLOYMENT") == "true"
        if is_streamlit_cloud:
            raise ValueError(
                "API keys are required. Please set your API keys in the sidebar before proceeding."
            )
        else:
            raise ValueError(
                "API keys are required. Please set SCRAPINGANT_API_KEY and EXA_API_KEY "
                "in your .env file or environment variables."
            )

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

@dataclass
class CompanyData:
    name: str
    domain: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    social_links: List[str] = None
    media_assets: List[MediaAsset] = None
    scrape_date: datetime = None
    product_lines: List[Dict[str, str]] = None
    competitors: List[Dict[str, str]] = None
    industry: Optional[str] = None
    key_metrics: Dict[str, str] = None
    technologies: List[str] = None

    def __post_init__(self):
        if self.social_links is None:
            self.social_links = []
        if self.media_assets is None:
            self.media_assets = []
        if self.product_lines is None:
            self.product_lines = []
        if self.competitors is None:
            self.competitors = []
        if self.technologies is None:
            self.technologies = []
        if self.key_metrics is None:
            self.key_metrics = {}
        if self.scrape_date is None:
            self.scrape_date = datetime.now()

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

class ScrapingAntScraper:
    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.base_url = "https://api.scrapingant.com/v2"
        self.usage_tracker = ApiUsageTracker()
        self.max_retries = max_retries
        
    def scrape_page(self, url: str) -> Optional[str]:
        start_time = datetime.now()
        retry_count = 0
        last_error = None

        # Validate URL format and accessibility
        try:
            # Try a HEAD request first to validate URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            if response.status_code == 404:
                raise ValueError(f"URL {url} returns 404 Not Found")
        except requests.exceptions.RequestException as e:
            logger.warning(f"URL validation failed: {str(e)}, proceeding with scraping attempt anyway")

        # First try with ScrapingAnt
        while retry_count < self.max_retries:
            try:
                session = requests.Session()
                session.max_redirects = 5
                adapter = requests.adapters.HTTPAdapter(
                    max_retries=3,
                    pool_maxsize=10,
                    pool_block=False
                )
                session.mount('http://', adapter)
                session.mount('https://', adapter)

                js_snippet = """
                    await new Promise(r => setTimeout(r, 2000));
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(r => setTimeout(r, 2000));
                    window.scrollTo(0, 0);
                """

                params = {
                    "url": url,
                    "x-api-key": self.api_key,
                    "browser": "true",
                    "proxy_type": "residential",
                    "cookies": "true",
                    "js_render": "true",
                    "wait_for_selector": "body",
                    "timeout": 30000,
                    "js_snippet": js_snippet.strip(),
                    "block_resources": "image,media,font",
                    "return_text": "true"
                }

                response = session.get(f"{self.base_url}/scrape", params=params)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    raise ValueError(f"ScrapingAnt API Error: {data['error']}")

                self.usage_tracker.log_usage(ApiUsage(
                    timestamp=datetime.now(),
                    api_name="ScrapingAnt",
                    endpoint="/scrape",
                    credits_used=1,
                    cost=0.0,
                    status="success",
                    response_time=(datetime.now() - start_time).total_seconds()
                ))

                return data.get("text") or data.get("content")

            except (requests.exceptions.RequestException, ValueError) as e:
                last_error = str(e)
                logger.error(f"ScrapingAnt Error: {last_error}")
                retry_count += 1
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        # If ScrapingAnt fails, try with regular requests as fallback
        logger.info(f"ScrapingAnt failed after {self.max_retries} attempts. Trying fallback method...")
        try:
            # Try different User-Agent strings
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1'
            ]
            
            session = requests.Session()
            session.max_redirects = 5
            
            for user_agent in user_agents:
                try:
                    headers = {'User-Agent': user_agent}
                    response = session.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    self.usage_tracker.log_usage(ApiUsage(
                        timestamp=datetime.now(),
                        api_name="ScrapingAnt",
                        endpoint="/scrape",
                        credits_used=0,
                        cost=0.0,
                        status="fallback_success",
                        response_time=(datetime.now() - start_time).total_seconds()
                    ))
                    
                    return response.text
                except requests.exceptions.RequestException:
                    continue
            
            raise ValueError(f"All fallback attempts failed for {url}")
            
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

    def extract_product_lines(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract product line information from the website"""
        product_lines = []
        
        # Common product section identifiers
        product_sections = soup.find_all(['section', 'div'], class_=lambda x: x and any(
            term in x.lower() for term in ['product', 'collection', 'category', 'featured']
        ))
        
        for section in product_sections:
            # Look for product names/titles
            product_titles = section.find_all(['h2', 'h3', 'h4'], class_=lambda x: x and any(
                term in x.lower() for term in ['title', 'name', 'product']
            ))
            
            # Look for product descriptions
            for title in product_titles:
                description = None
                desc_elem = title.find_next(['p', 'div'], class_=lambda x: x and any(
                    term in x.lower() for term in ['description', 'content', 'text']
                ))
                
                if desc_elem:
                    description = desc_elem.get_text(strip=True)
                
                product_lines.append({
                    'name': title.get_text(strip=True),
                    'description': description
                })
        
        return product_lines

    def extract_company_info(self, html_content: str, url: str) -> CompanyData:
        """Extract company information from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        domain = urlparse(url).netloc
        
        # Extract company name
        company_name = (
            soup.find('meta', property='og:site_name')['content'] if soup.find('meta', property='og:site_name') else
            soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else
            domain.split('.')[0]
        )
        
        # Extract description
        description = (
            soup.find('meta', property='og:description')['content'] if soup.find('meta', property='og:description') else
            soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else
            None
        )
        
        # Extract logo
        logo = soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')
        logo_url = urljoin(url, logo['href']) if logo else None
        
        # Extract social links
        social_links = []
        social_patterns = ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(pattern in href.lower() for pattern in social_patterns):
                social_links.append(href)
        
        # Extract media assets
        media_assets = []
        for img in soup.find_all('img', src=True):
            media_assets.append(MediaAsset(
                type='image',
                url=urljoin(url, img['src']),
                alt=img.get('alt', ''),
                metadata={'width': img.get('width'), 'height': img.get('height')}
            ))
        
        # Extract product lines
        product_lines = self.extract_product_lines(soup)
        
        return CompanyData(
            name=company_name,
            domain=domain,
            description=description,
            logo_url=logo_url,
            social_links=social_links,
            media_assets=media_assets,
            product_lines=product_lines
        )

class ExaSearcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.exa.ai/search"  # Updated endpoint
        self.usage_tracker = ApiUsageTracker()
    
    def search_company(self, company_name: str) -> Optional[Dict]:
        """Search for company information using Exa.ai's semantic search"""
        start_time = datetime.now()
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Remove special characters and extra spaces from company name
            clean_company_name = re.sub(r'[^\w\s-]', '', company_name).strip()
            
            query = {
                "query": f"company information about {clean_company_name}",
                "num_results": 5,
                "use_autoprompt": True,
                "type": "keyword",
                "include_domains": ["com", "org", "net", "co", "io", "ai"],  # Remove dots from domains
                "exclude_domains": ["wikipedia.org"]
            }
            
            logger.info(f"Searching Exa.ai for: {clean_company_name}")
            response = requests.post(
                self.base_url,
                headers=headers,
                json=query,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    raise ValueError(f"Exa.ai API Error: {data['error']}")
                
                # Log success
                self.usage_tracker.log_usage(ApiUsage(
                    timestamp=datetime.now(),
                    api_name="Exa.ai",
                    endpoint="/search",
                    credits_used=1,
                    cost=0.0,
                    status="success",
                    response_time=(datetime.now() - start_time).total_seconds()
                ))
                
                return data
            else:
                error_msg = f"Exa.ai Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                
                # Try without domain filters if we get a 400 error
                if response.status_code == 400 and "Invalid domain" in response.text:
                    logger.info("Retrying Exa.ai search without domain filters")
                    query.pop("include_domains", None)
                    query.pop("exclude_domains", None)
                    
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        json=query,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "error" in data:
                            raise ValueError(f"Exa.ai API Error: {data['error']}")
                        return data
                    
                raise ValueError(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request to Exa.ai failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in Exa.ai search: {str(e)}"
            logger.error(error_msg)
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
                scrape_date=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error extracting company info from Exa.ai results: {str(e)}")
            return None

def save_to_history(company_data: CompanyData):
    """Save company data to history CSV file"""
    # Define the columns we want to save
    columns = [
        'scrape_date', 'name', 'domain', 'description', 'logo_url',
        'social_links', 'media_assets', 'product_lines', 'competitors',
        'industry', 'key_metrics', 'technologies'
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
            'alt': str(asset.alt) if asset.alt else ''
        } for asset in company_data.media_assets]) if company_data.media_assets else '[]',
        'product_lines': str(company_data.product_lines) if company_data.product_lines else '[]',
        'competitors': str(company_data.competitors) if company_data.competitors else '[]',
        'industry': str(company_data.industry) if company_data.industry else '',
        'key_metrics': str(company_data.key_metrics) if company_data.key_metrics else '{}',
        'technologies': str(company_data.technologies) if company_data.technologies else '[]'
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
        
        # Convert timestamps back to string format before saving
        df['scrape_date'] = df['scrape_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
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
        
        exa_searcher = ExaSearcher(EXA_API_KEY)
        search_results = exa_searcher.search_company(scraped_data.name)
        exa_data = exa_searcher.extract_company_info(search_results) if search_results else None
        
        technologies = extract_technologies(soup)
        key_metrics = extract_key_metrics(soup, scraped_data.description)
        
        industry = None
        if exa_data and exa_data.description:
            industry_keywords = ["technology", "software", "healthcare", "retail", "finance", "manufacturing", "automotive", "energy", "telecommunications", "education", "real estate", "construction", "agriculture"]
            description_lower = exa_data.description.lower()
            for keyword in industry_keywords:
                if keyword in description_lower:
                    industry = keyword.title()
                    break
        
        competitors = []
        if industry:
            competitors = find_competitors(scraped_data.name, industry, exa_searcher)
        
        if exa_data:
            if exa_data.description and (not scraped_data.description or len(exa_data.description) > len(scraped_data.description)):
                scraped_data.description = exa_data.description
            if not scraped_data.name or scraped_data.name == scraped_data.domain.split('.')[0]:
                scraped_data.name = exa_data.name
        
        scraped_data.technologies = technologies
        scraped_data.key_metrics = key_metrics
        scraped_data.competitors = competitors
        scraped_data.industry = industry
        
        metrics = {
            "social_presence_score": len(scraped_data.social_links) / 4.0 * 100,
            "media_richness_score": min(len(scraped_data.media_assets) / 10 * 100, 100),
            "information_completeness": sum(1 for v in [scraped_data.name, scraped_data.description, scraped_data.logo_url, scraped_data.social_links, scraped_data.technologies, scraped_data.competitors] if v) / 6.0 * 100
        }
        
        save_to_history(scraped_data)
        
        company_dict = {
            "name": scraped_data.name,
            "domain": scraped_data.domain,
            "description": scraped_data.description,
            "logo_url": scraped_data.logo_url,
            "social_links": scraped_data.social_links,
            "media_assets": [{
                "type": asset.type,
                "url": asset.url,
                "alt": asset.alt,
                "metadata": asset.metadata
            } for asset in scraped_data.media_assets],
            "product_lines": scraped_data.product_lines,
            "competitors": scraped_data.competitors,
            "industry": scraped_data.industry,
            "key_metrics": scraped_data.key_metrics,
            "technologies": scraped_data.technologies,
            "scrape_date": scraped_data.scrape_date.isoformat()
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
        search_results = searcher.search_company(company_name)
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
    """Find competitors using Exa.ai's semantic search"""
    try:
        # Construct a detailed search query for competitors
        query = f"top competitors of {company_name} in {industry} industry"
        search_results = exa_searcher.search_company(query)
        
        if not search_results or "results" not in search_results:
            return []
        
        competitors = []
        seen_companies = set()
        
        for result in search_results["results"]:
            text = result.get("text", "")
            
            # Look for competitor mentions
            if "competitor" in text.lower() or "rival" in text.lower():
                # Extract company names using common patterns
                lines = text.split(".")
                for line in lines:
                    if "competitor" in line.lower() or "rival" in line.lower():
                        # Clean and extract competitor info
                        comp_info = {
                            "name": "",
                            "description": line.strip()
                        }
                        
                        # Try to extract specific company name
                        words = line.split()
                        for i, word in enumerate(words):
                            if word.lower() in ["is", "are", "includes", "such", "like"]:
                                if i + 1 < len(words):
                                    comp_name = words[i + 1]
                                    if comp_name not in seen_companies:
                                        comp_info["name"] = comp_name
                                        seen_companies.add(comp_name)
                                        competitors.append(comp_info)
                                        break
        
        return competitors[:5]  # Return top 5 competitors
        
    except Exception as e:
        logger.error(f"Error finding competitors: {str(e)}")
        return []

def batch_analyze_companies(urls: List[str], max_retries: int = 3, delay_between_requests: int = 2) -> Dict[str, Dict]:
    """
    Analyze multiple companies in batch with retry logic and rate limiting
    """
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
            
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {str(e)}")
            results[url] = {
                "success": False,
                "error": str(e)
            }
    
    # Generate summary
    successful = sum(1 for r in results.values() if r.get("success", False))
    logger.info(f"Batch analysis completed. Success: {successful}/{total_urls}")
    
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