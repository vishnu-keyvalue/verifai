#!/usr/bin/env python3
"""
Alternative search APIs as fallback when Google API quota is exceeded
"""

import aiohttp
import asyncio
import logging
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class AlternativeSearchAPIs:
    """Alternative search APIs for when Google API quota is exceeded."""
    
    def __init__(self):
        self.bing_api_key = os.getenv('BING_API_KEY')
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.duckduckgo_api_key = os.getenv('DUCKDUCKGO_API_KEY')
    
    async def search_with_fallback(self, query: str, num_results: int = 5) -> List[Dict]:
        """Try multiple search APIs in order of preference."""
        
        # Try Bing Search API first (if configured)
        if self.bing_api_key:
            try:
                results = await self._search_bing(query, num_results)
                if results:
                    logging.info(f"Bing search successful: {len(results)} results")
                    return results
            except Exception as e:
                logging.warning(f"Bing search failed: {e}")
        
        # Try SerpAPI (if configured)
        if self.serpapi_key:
            try:
                results = await self._search_serpapi(query, num_results)
                if results:
                    logging.info(f"SerpAPI search successful: {len(results)} results")
                    return results
            except Exception as e:
                logging.warning(f"SerpAPI search failed: {e}")
        
        # Try DuckDuckGo (if configured)
        if self.duckduckgo_api_key:
            try:
                results = await self._search_duckduckgo(query, num_results)
                if results:
                    logging.info(f"DuckDuckGo search successful: {len(results)} results")
                    return results
            except Exception as e:
                logging.warning(f"DuckDuckGo search failed: {e}")
        
        # If all APIs fail, return empty results
        logging.warning("All alternative search APIs failed")
        return []
    
    async def _search_bing(self, query: str, num_results: int) -> List[Dict]:
        """Search using Bing Web Search API."""
        if not self.bing_api_key:
            return []
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": self.bing_api_key
        }
        params = {
            "q": query,
            "count": min(num_results, 10),
            "mkt": "en-US"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    web_pages = data.get('webPages', {}).get('value', [])
                    
                    results = []
                    for page in web_pages:
                        results.append({
                            'title': page.get('name', ''),
                            'link': page.get('url', ''),
                            'snippet': page.get('snippet', '')
                        })
                    
                    return results
                else:
                    raise Exception(f"Bing API error: {response.status}")
    
    async def _search_serpapi(self, query: str, num_results: int) -> List[Dict]:
        """Search using SerpAPI (Google search results)."""
        if not self.serpapi_key:
            return []
        
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": min(num_results, 10)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    organic_results = data.get('organic_results', [])
                    
                    results = []
                    for result in organic_results:
                        results.append({
                            'title': result.get('title', ''),
                            'link': result.get('link', ''),
                            'snippet': result.get('snippet', '')
                        })
                    
                    return results
                else:
                    raise Exception(f"SerpAPI error: {response.status}")
    
    async def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict]:
        """Search using DuckDuckGo Instant Answer API."""
        if not self.duckduckgo_api_key:
            return []
        
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # DuckDuckGo returns different types of results
                    if data.get('Abstract'):
                        results.append({
                            'title': data.get('Heading', ''),
                            'link': data.get('AbstractURL', ''),
                            'snippet': data.get('Abstract', '')
                        })
                    
                    # Add related topics
                    for topic in data.get('RelatedTopics', [])[:num_results-1]:
                        if isinstance(topic, dict) and topic.get('Text'):
                            results.append({
                                'title': topic.get('Text', '')[:50] + '...',
                                'link': topic.get('FirstURL', ''),
                                'snippet': topic.get('Text', '')
                            })
                    
                    return results
                else:
                    raise Exception(f"DuckDuckGo API error: {response.status}")

# Usage example
async def test_alternative_search():
    """Test the alternative search APIs."""
    searcher = AlternativeSearchAPIs()
    
    # Check which APIs are configured
    print("Configured APIs:")
    print(f"Bing: {'Yes' if searcher.bing_api_key else 'No'}")
    print(f"SerpAPI: {'Yes' if searcher.serpapi_key else 'No'}")
    print(f"DuckDuckGo: {'Yes' if searcher.duckduckgo_api_key else 'No'}")
    
    if not any([searcher.bing_api_key, searcher.serpapi_key, searcher.duckduckgo_api_key]):
        print("\nNo alternative APIs configured!")
        print("To use alternative APIs, add these to your .env file:")
        print("BING_API_KEY=your_bing_key")
        print("SERPAPI_KEY=your_serpapi_key")
        print("DUCKDUCKGO_API_KEY=your_duckduckgo_key")
        return
    
    # Test search
    results = await searcher.search_with_fallback("Argentina World Cup 2022", 3)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']}")
        print(f"   URL: {result['link']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
        print()

if __name__ == "__main__":
    asyncio.run(test_alternative_search()) 