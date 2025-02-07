# Media Asset Scraper

A robust Python tool for scraping and analyzing company data, media assets, and social presence from websites.

## Features

- Website scraping with JavaScript support using ScrapingAnt API
- Company information extraction (name, description, logo, social links)
- Media asset detection and analysis
- API usage tracking and analytics
- Comparison of different scraping approaches (ScrapingAnt vs Exa.ai)
- CSV export for historical data tracking

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone git@github-personal:YOUR_USERNAME/media-asset-scrape.git
cd media-asset-scrape
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys:
# SCRAPINGANT_API_KEY=your_key_here
# EXA_API_KEY=your_key_here
```

## Usage

### Basic Company Analysis
```python
from company_data_comparison import analyze_company_data

results = analyze_company_data("https://example.com")
if results["success"]:
    print(f"Company Name: {results['company_data']['name']}")
    print(f"Social Presence Score: {results['metrics']['social_presence_score']}%")
```

### Compare Scraping Approaches
```python
from company_data_comparison import compare_approaches

comparison = compare_approaches("https://example.com", "Example Company")
```

### Track API Usage
```python
from company_data_comparison import get_api_usage_stats

stats = get_api_usage_stats()
print(f"Total API Requests: {stats['stats']['total_requests']}")
```

## Data Output

The tool generates two main CSV files:
- `company_data_history.csv`: Historical data of analyzed companies
- `api_usage.csv`: API usage tracking and costs

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ScrapingAnt](https://scrapingant.com/) for website scraping
- [Exa.ai](https://exa.ai/) for semantic search capabilities 