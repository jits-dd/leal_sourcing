import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from firecrawl.firecrawl import FirecrawlApp
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
load_dotenv()
from typing import Type, List, Dict, Any

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")

# Change to localhost:3002 if running locally
# firecrawl = FirecrawlApp(api_url="http://localhost:3002", api_key="123")

# Uncomment for cloud version
firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

serper_headers = {
    "X-API-KEY": os.getenv("SERPER_API_KEY"),
    "Content-Type": "application/json",
}

class LeadFinderInput(BaseModel):
    query: str = Field(description="Search query to look up leads (e.g., pet food, pet grooming)")
    location: str = Field(description="Location of the query (e.g., India)")
    industry_keywords: List[str] = Field(description="Keywords related to the industry (e.g., pet food, pet grooming)")
    relevance_keywords: List[str] = Field(description="Keywords related to relevance (e.g., acquisition, investment)")

class LeadFinderTool(BaseTool):
    name: str = "LeadFinderTool"
    description: str = (
        "Find companies in a specific industry and location that are actively acquiring or open to acquisitions. "
        "Use the query input parameter to search for the niche, and the location for the specific location."
    )
    args_schema: Type[BaseModel] = LeadFinderInput

    def _run(self, query: str, location: str, industry_keywords: List[str], relevance_keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search the web for companies in the specified industry and location.
        """
        # Search for companies using Serper API
        search_results = self._search_web(query, location)

        # Extract and enrich company data
        companies = []
        for result in search_results:
            company_data = self._extract_company_data(result, industry_keywords, relevance_keywords)
            if company_data:
                companies.append(company_data)

        # Rank companies by relevance
        ranked_companies = self._rank_companies(companies)
        return ranked_companies
    
    def _search_web(self, query: str, location: str) -> List[Dict[str, Any]]:
        """
        Search the web for companies using Serper API.
        """
        request_body = {
            "q": f"{query} companies in {location}",
            "gl": location.lower(),  # Geographic location
            "hl": "en",  # Language
            "num": 10,  # Number of results
        }
        response = requests.post(
            url="https://google.serper.dev/search",
            headers = {"X-API-KEY": os.getenv("SERPER_API_KEY"), "Content-Type": "application/json"},
            json=request_body,
        )
        return response.json().get("organic", [])

    def _extract_company_data(self, search_result: Dict[str, Any], industry_keywords: List[str], relevance_keywords: List[str]) -> Dict[str, Any]:
        """
        Extract and enrich company data from search results.
        """
        company_name = search_result.get("title", "")
        website = search_result.get("link", "")
        snippet = search_result.get("snippet", "")

        # Scrape the company website for additional details
        scraped_data = self._scrape_website(website)

        # Determine strategic fit and relevance dynamically
        strategic_fit = self._calculate_strategic_fit(snippet, industry_keywords)
        relevance = self._calculate_relevance(snippet, relevance_keywords)

        return {
            "company_name": company_name,
            "website": website,
            "industry": "Pet Care",  # Placeholder, can be extracted from snippet
            "acquisition_history": "Not found",  # Placeholder, can be extracted from snippet
            "strategic_fit": strategic_fit,
            "relevance": relevance,
            **scraped_data,  # Include scraped data
        }
    
    def _calculate_strategic_fit(self, snippet: str, industry_keywords: List[str]) -> str:
        """
        Calculate strategic fit based on the presence of industry keywords in the snippet.
        """
        keyword_count = sum(1 for keyword in industry_keywords if keyword in snippet.lower())
        if keyword_count >= 3:  # Adjust threshold as needed
            return "High"
        elif keyword_count >= 1:
            return "Medium"
        else:
            return "Low"
        
    def _calculate_relevance(self, snippet: str, relevance_keywords: List[str]) -> str:
        """
        Calculate relevance based on the presence of relevance keywords in the snippet.
        """
        keyword_count = sum(1 for keyword in relevance_keywords if keyword in snippet.lower())
        if keyword_count >= 2:  # Adjust threshold as needed
            return "High"
        elif keyword_count >= 1:
            return "Medium"
        else:
            return "Low"

    def _scrape_website(self, url: str) -> Dict[str, Any]:
        """
        Scrape a company's website for additional details.
        """
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract relevant data from the website
            company_name = soup.title.string if soup.title else "Not found"
            mission = self._extract_mission(soup)

            return {
                "company_name": company_name,
                "company_mission": mission,
            }
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return {}


    def _extract_mission(self, soup: BeautifulSoup) -> str:
        """
        Extract company mission from the website.
        """
        # Look for common mission statement tags
        mission = soup.find("h2", string=lambda text: "mission" in text.lower())
        return mission.get_text().strip() if mission else "Not found"

    def _rank_companies(self, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank companies by relevance, strategic fit, and acquisition history.
        """
        ranked_companies = sorted(
            companies,
            key=lambda x: (
                x.get("relevance", "Medium"),
                x.get("strategic_fit", "Medium"),
            ),
            reverse=True,
        )
        return ranked_companies
    

class ExtractionSchema(BaseModel):
    company_name: str
    news: str
    company_mission: str

class LeadExtractorInput(BaseModel):
    url: str = Field(description="Url to extract data from")

class LeadExtractorTool(BaseTool):
    name: str = "LeadExtractor"
    description: str = "A tool for extracting lead information from a given url."
    args_schema: Type[BaseModel] = LeadExtractorInput

    def _run(self, url: str) -> Dict[str, Any]:
        """
        Extract and enrich data from a company's website.
        """
        return self._scrape_website(url)