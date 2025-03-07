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
        Search the web for companies in the specified industry and location using Perplexity API.
        """
        # Prepare the prompt for Perplexity API
        prompt = f"Find companies in {location} that are in the {query} industry and are actively acquiring or open to acquisitions. Provide details such as name, industry, acquisition history, and strategic fit."

        # Call Perplexity API
        response = llm.chat.completions.create(
            model="sonar-pro",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        # Extract and parse the response
        companies = self._parse_response(response.choices[0].message.content)
        return companies

    def _parse_response(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Parse the response from Perplexity API to extract company details.
        """
        # This is a placeholder for parsing logic. You need to implement this based on the actual response format.
        # For example, if the response is a list of companies in JSON format, you can parse it accordingly.
        companies = []
        # Example parsing logic (adjust based on actual response format)
        for line in response_content.split("\n"):
            if "company_name" in line:
                company_data = {
                    "company_name": line.split(":")[1].strip(),
                    "industry": "Pet Care",  # Placeholder, can be extracted from response
                    "acquisition_history": "Not found",  # Placeholder, can be extracted from response
                    "strategic_fit": "High",  # Placeholder, can be extracted from response
                    "relevance": "High",  # Placeholder, can be extracted from response
                }
                companies.append(company_data)
        return companies

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