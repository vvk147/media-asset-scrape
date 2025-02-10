#!/usr/bin/env python3

import json
import logging
import os
from dotenv import load_dotenv
from company_data_comparison import batch_analyze_companies, set_api_keys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'batch_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
    
    # Initialize API keys from environment
    set_api_keys()
    
    # Load URLs from file
    with open('company_urls.json', 'r') as f:
        urls = json.load(f)
    
    logger.info(f"Loaded {len(urls)} URLs for analysis")
    
    # Run batch analysis
    results = batch_analyze_companies(
        urls,
        max_retries=3,
        delay_between_requests=2  # 2 seconds between requests to avoid rate limiting
    )
    
    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_urls': len(urls),
        'successful': sum(1 for r in results.values() if r.get('success', False)),
        'failed': sum(1 for r in results.values() if not r.get('success', False)),
        'results': results
    }
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save summary to output directory
    output_file = os.path.join('output', f'batch_analysis_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Analysis complete. Success rate: {summary['successful']}/{summary['total_urls']}")
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 