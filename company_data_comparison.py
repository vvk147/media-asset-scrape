import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import requests
import json
from urllib.parse import urlparse, urljoin
import logging
import csv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
SCRAPINGANT_API_KEY = os.getenv("SCRAPINGANT_API_KEY", "c122e752650a4f5da79dd7b65be45ef8")
EXA_API_KEY = os.getenv("EXA_API_KEY", "ae83f8f2-1894-4b47-9df3-08b1d4c15cc5")

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

    def __post_init__(self):
        if self.social_links is None:
            self.social_links = []
        if self.media_assets is None:
            self.media_assets = []
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
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.scrapingant.com/v2"
        self.usage_tracker = ApiUsageTracker()
        
    def scrape_page(self, url: str) -> Optional[str]:
        start_time = datetime.now()
        try:
            params = {
                "url": url,
                "x-api-key": self.api_key,
                "browser": "true"
            }
            response = requests.get(f"{self.base_url}/general", params=params)
            
            # Track API usage
            usage = ApiUsage(
                timestamp=start_time,
                api_name="ScrapingAnt",
                endpoint="/general",
                credits_used=1,  # 1 credit per request
                cost=0.0002,  # $0.0002 per credit
                status=str(response.status_code),
                response_time=(datetime.now() - start_time).total_seconds()
            )
            self.usage_tracker.log_usage(usage)
            
            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"ScrapingAnt Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"ScrapingAnt Exception: {str(e)}")
            return None

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
        
        return CompanyData(
            name=company_name,
            domain=domain,
            description=description,
            logo_url=logo_url,
            social_links=social_links,
            media_assets=media_assets
        )

def save_to_csv(company_data: CompanyData, filename: str = "company_data_history.csv"):
    """Save company data to CSV file"""
    flat_data = {
        'company_name': company_data.name,
        'domain': company_data.domain,
        'description': company_data.description,
        'logo_url': company_data.logo_url,
        'social_links': ','.join(company_data.social_links),
        'media_asset_count': len(company_data.media_assets),
        'scrape_date': company_data.scrape_date.isoformat()
    }
    
    file_exists = Path(filename).exists()
    
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_data)
    
    logger.info(f"Data saved to {filename}")

def analyze_company_data(company_url: str) -> Dict:
    """Analyze company data from the website"""
    
    results = {
        "success": False,
        "company_data": None,
        "metrics": {},
        "error": None
    }
    
    logger.info(f"Analyzing company data for: {company_url}")
    
    try:
        scraper = ScrapingAntScraper(SCRAPINGANT_API_KEY)
        html_content = scraper.scrape_page(company_url)
        
        if html_content:
            company_data = scraper.extract_company_info(html_content, company_url)
            
            # Calculate metrics
            metrics = {
                "social_presence_score": len(company_data.social_links) / 4.0 * 100,
                "media_richness_score": min(len(company_data.media_assets) / 10 * 100, 100),
                "information_completeness": sum(
                    1 for v in [
                        company_data.name, company_data.description, 
                        company_data.logo_url, company_data.social_links
                    ] if v
                ) / 4.0 * 100
            }
            
            results.update({
                "success": True,
                "company_data": company_data.__dict__,
                "metrics": metrics
            })
            
            save_to_csv(company_data)
            
        else:
            results["error"] = "Failed to scrape website content"
            
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