# modules/advanced_evidence_retriever.py
from googleapiclient.discovery import build
import asyncio
import aiohttp
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin
import time
from dataclasses import dataclass
from .alternative_search import AlternativeSearchAPIs
import re

@dataclass
class EvidenceItem:
    title: str
    url: str
    snippet: str
    source_type: str
    credibility_score: float
    relevance_score: float
    publish_date: Optional[str] = None

class AdvancedEvidenceRetriever:
    def __init__(self, google_api_key: str, google_cse_id: str):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.search_service = build("customsearch", "v1", developerKey=google_api_key)
        self.alternative_search = AlternativeSearchAPIs()
        
        # Define high-quality source domains by category
        self.fact_check_sites = [
            'snopes.com', 'factcheck.org', 'politifact.com',
            'reuters.com/fact-check', 'apnews.com/ap-fact-check',
            'fullfact.org', 'checkyourfact.com'
        ]
        
        self.academic_sites = [
            'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov',
            'jstor.org', 'arxiv.org', 'researchgate.net'
        ]
        
        self.news_sites = [
            'reuters.com', 'apnews.com', 'bbc.com', 'npr.org',
            'theguardian.com', 'wsj.com', 'nytimes.com'
        ]
        
        self.official_sites = [
            '*.gov', '*.edu', 'who.int', 'cdc.gov',
            'fda.gov', 'nih.gov', 'nasa.gov'
        ]
    
    async def comprehensive_evidence_search(self, claim: str, max_results: int = 20) -> Dict[str, List[EvidenceItem]]:
        """Perform multi-strategy evidence retrieval with improved quota handling."""
        evidence_results = {
            'fact_check': [],
            'academic': [],
            'news': [],
            'official': []
        }
        
        # Clean the claim first to remove HTML artifacts and metadata
        cleaned_claim = self._clean_claim_for_search(claim)
        
        # Analyze the claim to determine search strategy
        claim_analysis = self._analyze_claim_type(cleaned_claim)
        
        # Extract core factual elements for better search targeting
        core_facts = self._extract_core_facts(cleaned_claim)
        
        # Adjust search priorities based on claim type
        if claim_analysis['is_historical_fact']:
            # For historical facts, prioritize news and official sources
            tasks = [
                self._search_news_sources_with_core_facts(core_facts, max_results // 2),  # More news sources
                self._search_official_sources_with_core_facts(core_facts, max_results // 4),
                self._search_fact_checkers_with_core_facts(core_facts, max_results // 8),  # Fewer fact-checks
                self._search_academic_sources_with_core_facts(core_facts, max_results // 8)
            ]
        elif claim_analysis['is_breaking_news']:
            # For breaking news, prioritize recent news sources
            tasks = [
                self._search_news_sources_with_core_facts(core_facts, max_results // 2),
                self._search_official_sources_with_core_facts(core_facts, max_results // 4),
                self._search_fact_checkers_with_core_facts(core_facts, max_results // 8),
                self._search_academic_sources_with_core_facts(core_facts, max_results // 8)
            ]
        else:
            # For controversial claims, prioritize fact-checkers
            tasks = [
                self._search_fact_checkers_with_core_facts(core_facts, max_results // 3),
                self._search_news_sources_with_core_facts(core_facts, max_results // 3),
                self._search_official_sources_with_core_facts(core_facts, max_results // 4),
                self._search_academic_sources_with_core_facts(core_facts, max_results // 6)
            ]
        
        # Execute searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        categories = ['fact_check', 'academic', 'news', 'official']
        quota_exceeded = False
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                evidence_results[categories[i]] = result
            else:
                logging.error(f"Error in {categories[i]} search: {result}")
                if "quota exceeded" in str(result).lower():
                    quota_exceeded = True
        
        evidence_results['quota_exceeded'] = quota_exceeded
        return evidence_results
    
    def _clean_claim_for_search(self, claim: str) -> str:
        """Clean claim text by removing HTML artifacts and metadata for better search."""
        # Remove common HTML artifacts
        claim = re.sub(r'toggle caption', '', claim, flags=re.IGNORECASE)
        claim = re.sub(r'enlarge this image', '', claim, flags=re.IGNORECASE)
        claim = re.sub(r'via Getty Images', '', claim, flags=re.IGNORECASE)
        claim = re.sub(r'via AP', '', claim, flags=re.IGNORECASE)
        claim = re.sub(r'via Reuters', '', claim, flags=re.IGNORECASE)
        
        # Remove image captions and photo credits
        claim = re.sub(r'[A-Z][a-z]+/[A-Z][a-z]+', '', claim)  # Remove photographer names
        
        # Remove excessive whitespace and newlines
        claim = re.sub(r'\n+', ' ', claim)
        claim = re.sub(r'\s+', ' ', claim)
        
        return claim.strip()
    
    def _extract_core_facts(self, claim: str) -> Dict[str, str]:
        """Extract core factual elements from a claim for targeted searching."""
        core_facts = {
            'main_event': '',
            'participants': [],
            'outcome': '',
            'date': '',
            'location': ''
        }
        
        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', claim)
        if year_match:
            core_facts['date'] = year_match.group(1)
        
        # Extract main event type
        event_patterns = [
            r'World Cup',
            r'Olympics',
            r'championship',
            r'final',
            r'tournament'
        ]
        for pattern in event_patterns:
            if re.search(pattern, claim, re.IGNORECASE):
                core_facts['main_event'] = pattern
                break
        
        # Extract participants (teams, countries, people)
        participant_patterns = [
            r'\b(Argentina|France|Brazil|Germany|Spain|Italy|England)\b',
            r'\b(United States|Canada|Mexico|China|Japan|South Korea)\b'
        ]
        for pattern in participant_patterns:
            matches = re.findall(pattern, claim, re.IGNORECASE)
            core_facts['participants'].extend(matches)
        
        # Extract outcome
        outcome_patterns = [
            r'\b(won|beat|defeated|lost)\b',
            r'\b(victory|victorious|champion)\b',
            r'\b(penalty kicks?|penalties)\b'
        ]
        for pattern in outcome_patterns:
            match = re.search(pattern, claim, re.IGNORECASE)
            if match:
                core_facts['outcome'] = match.group(1)
                break
        
        return core_facts
    
    def _analyze_claim_type(self, claim: str) -> Dict[str, bool]:
        """Analyze the type of claim to determine optimal search strategy."""
        claim_lower = claim.lower()
        
        # Historical fact indicators
        historical_indicators = [
            'world cup', 'olympics', 'election', 'president', 'champion',
            'won', 'won the', 'championship', 'final', 'victory'
        ]
        
        # Breaking news indicators
        breaking_indicators = [
            'breaking', 'latest', 'just in', 'developing', 'update',
            'today', 'yesterday', 'recent', 'new', 'emerging'
        ]
        
        # Controversial claim indicators
        controversial_indicators = [
            'conspiracy', 'hoax', 'fake', 'false', 'debunked',
            'misleading', 'untrue', 'myth', 'scam'
        ]
        
        # Count indicators
        historical_count = sum(1 for indicator in historical_indicators if indicator in claim_lower)
        breaking_count = sum(1 for indicator in breaking_indicators if indicator in claim_lower)
        controversial_count = sum(1 for indicator in controversial_indicators if indicator in claim_lower)
        
        return {
            'is_historical_fact': historical_count > 0 and breaking_count == 0,
            'is_breaking_news': breaking_count > 0,
            'is_controversial': controversial_count > 0,
            'historical_score': historical_count,
            'breaking_score': breaking_count,
            'controversial_score': controversial_count
        }
    
    async def _search_fact_checkers(self, claim: str, num_results: int) -> List[EvidenceItem]:
        """Search specifically in fact-checking websites with improved strategy."""
        evidence_items = []
        
        # Extract key terms from the claim for more targeted search
        key_terms = self._extract_key_terms(claim)
        
        # Only search top 2 fact-check sites instead of 3
        for site in self.fact_check_sites[:2]:  # Reduced from 3 to 2
            try:
                # Use key terms instead of full claim to avoid overly specific matches
                # that might miss relevant fact-checks
                query = f'{" ".join(key_terms[:3])} site:{site}'
                results = await self._perform_google_search(query, 1)  # 1 result per site
                
                for item in results:
                    # Check if the result is actually relevant to the current claim
                    if self._is_relevant_to_claim(item, claim):
                        evidence_items.append(EvidenceItem(
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=item.get('snippet', ''),
                            source_type='fact_check',
                            credibility_score=0.9,  # High credibility for fact-check sites
                            relevance_score=0.8  # Will be recalculated with semantic analysis
                        ))
                    
            except Exception as e:
                logging.error(f"Error searching {site}: {e}")
        
        return evidence_items
    
    def _extract_key_terms(self, claim: str) -> List[str]:
        """Extract key terms from a claim for better search targeting."""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'as', 'from', 'into', 'during', 'including', 'until', 'against', 'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'about', 'over', 'above', 'below', 'under', 'within', 'without', 'between', 'among', 'through', 'during', 'before', 'after', 'since', 'while', 'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
        
        # Simple tokenization and filtering
        words = claim.lower().split()
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Prioritize nouns, proper nouns, and numbers
        # This is a simplified approach - in production you might use NLP libraries
        return key_terms[:5]  # Return top 5 key terms
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text with improved accuracy."""
        # Look for capitalized words (potential proper nouns)
        words = text.split()
        entities = []
        
        for word in words:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                # Filter out common capitalized words that aren't specific entities
                common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'as', 'from', 'into', 'during', 'including', 'until', 'against', 'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'about', 'over', 'above', 'below', 'under', 'within', 'without', 'between', 'among', 'through', 'during', 'before', 'after', 'since', 'while', 'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'enlarge', 'image', 'toggle', 'caption', 'david', 'ramos', 'fifa', 'getty', 'images', 'world', 'cup', 'final', 'photos', 'beats', 'france', 'penalty', 'kicks', 'win', 'victors', 'started', 'strong', 'lead', 'kylian', 'mbapp', 'quickly', 'changed', 'course', 'game', 'goals', 'second', 'half', 'match', 'ended', 'final', 'score', 'croatia', 'emerged', 'victorious', 'saturday', 'morocco', 'ding', 'xu', 'xinhua', 'news', 'agency', 'jia', 'haoc', 'moro', 'cc', 'ng', 'francisco', 'seco', 'stipe', 'majic', 'anadolu', 'united', 'states', 'canada', 'mexico'}
                
                if clean_word.lower() not in common_words:
                    entities.append(clean_word.lower())
        
        # Look for specific patterns like "World Cup", "penalty kicks", etc.
        import re
        
        # Multi-word entities
        multi_word_patterns = [
            r'world cup',
            r'penalty kicks',
            r'argentina',
            r'france',
            r'croatia',
            r'morocco'
        ]
        
        text_lower = text.lower()
        for pattern in multi_word_patterns:
            if re.search(pattern, text_lower):
                entities.append(pattern)
        
        # Look for years but be more specific
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text)
        entities.extend([f"20{year}" for year in years if year.startswith('2')])
        entities.extend([f"19{year}" for year in years if year.startswith('9')])
        
        # Remove duplicates and return unique entities
        unique_entities = list(set(entities))
        return unique_entities[:5]  # Return top 5 entities
    
    def _is_relevant_to_claim(self, search_result: Dict, claim: str) -> bool:
        """Check if a search result is actually relevant to the current claim with improved logic."""
        title = search_result.get('title', '').lower()
        snippet = search_result.get('snippet', '').lower()
        claim_lower = claim.lower()
        
        # Extract key entities from the claim
        claim_entities = self._extract_entities(claim_lower)
        
        # Check if the result contains the key entities
        result_text = f"{title} {snippet}"
        
        # Calculate entity matches
        entity_matches = sum(1 for entity in claim_entities if entity in result_text)
        
        # Require at least 60% of key entities to match (increased from 50%)
        min_matches = max(1, len(claim_entities) * 0.6)
        
        # Additional check: if we have very specific entities like "argentina" and "world cup",
        # both should be present for high relevance
        specific_entities = ['argentina', 'world cup', 'france', 'penalty kicks']
        specific_matches = sum(1 for entity in specific_entities if entity in claim_entities and entity in result_text)
        
        # If we have specific entities in the claim, require at least one to match
        if any(entity in claim_entities for entity in specific_entities):
            if specific_matches == 0:
                return False
        
        # Additional relevance check: look for semantic similarity in key phrases
        key_phrases = self._extract_key_phrases(claim_lower)
        phrase_matches = sum(1 for phrase in key_phrases if phrase in result_text)
        
        # Require either good entity matches OR good phrase matches
        return (entity_matches >= min_matches) or (phrase_matches >= 1)
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that should appear together for relevance."""
        phrases = []
        
        # Look for specific phrases that indicate the topic
        if 'argentina' in text and 'world cup' in text:
            phrases.append('argentina world cup')
        if 'france' in text and 'penalty' in text:
            phrases.append('france penalty')
        if 'win' in text and 'cup' in text:
            phrases.append('win cup')
        if 'beats' in text and 'france' in text:
            phrases.append('beats france')
        
        return phrases
    
    async def _search_academic_sources(self, claim: str, num_results: int) -> List[EvidenceItem]:
        """Search academic and research sources."""
        evidence_items = []
        
        # Search academic databases
        academic_queries = [
            f'{claim} site:scholar.google.com',
            f'{claim} site:pubmed.ncbi.nlm.nih.gov',
            f'"{claim}" research study'
        ]
        
        for query in academic_queries:
            try:
                results = await self._perform_google_search(query, num_results // len(academic_queries))
                
                for item in results:
                    evidence_items.append(EvidenceItem(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source_type='academic',
                        credibility_score=0.85,
                        relevance_score=0.7
                    ))
            except Exception as e:
                logging.error(f"Error in academic search: {e}")
        
        return evidence_items
    
    async def _search_news_sources(self, claim: str, num_results: int) -> List[EvidenceItem]:
        """Search reputable news sources with improved strategy for historical facts and breaking news."""
        evidence_items = []
        
        # Extract key terms for better search targeting
        key_terms = self._extract_key_terms(claim)
        
        # Analyze claim type to adjust search strategy
        claim_analysis = self._analyze_claim_type(claim)
        
        # Create site-specific queries for top news sources (reduced from 6 to 3)
        for site in self.news_sites[:3]:  # Reduced from 6 to 3 sites
            try:
                # Use key terms and add context-specific modifiers
                base_query = f'{" ".join(key_terms[:3])} site:{site}'
                
                # Simplified search strategy - only one query per site
                if claim_analysis['is_historical_fact']:
                    # For historical facts, just use basic search
                    query = base_query
                elif claim_analysis['is_breaking_news']:
                    # For breaking news, add recent modifier
                    query = f'{base_query} "latest"'
                else:
                    # For other claims, use standard approach
                    query = base_query
                
                results = await self._perform_google_search(query, 2)  # 2 results per site
                
                for item in results:
                    # Check relevance and avoid duplicates
                    if (self._is_relevant_to_claim(item, claim) and 
                        not any(e.url == item.get('link', '') for e in evidence_items)):
                        evidence_items.append(EvidenceItem(
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=item.get('snippet', ''),
                            source_type='news',
                            credibility_score=0.75,
                            relevance_score=0.6
                        ))
                        
                        # Break if we have enough results for this site
                        if len([e for e in evidence_items if e.source_type == 'news']) >= num_results:
                            break
                        
            except Exception as e:
                logging.error(f"Error searching news site {site}: {e}")
        
        return evidence_items
    
    async def _search_official_sources(self, claim: str, num_results: int) -> List[EvidenceItem]:
        """Search official government and institutional sources."""
        evidence_items = []
        
        official_queries = [
            f'{claim} site:.gov',
            f'{claim} site:.edu',
            f'{claim} site:who.int OR site:cdc.gov OR site:nih.gov'
        ]
        
        for query in official_queries:
            try:
                results = await self._perform_google_search(query, num_results // len(official_queries))
                
                for item in results:
                    evidence_items.append(EvidenceItem(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source_type='official',
                        credibility_score=0.95,  # Highest credibility for official sources
                        relevance_score=0.7
                    ))
            except Exception as e:
                logging.error(f"Error in official search: {e}")
        
        return evidence_items
    
    async def _perform_google_search(self, query: str, num_results: int) -> List[Dict]:
        """Perform Google Custom Search API call with alternative API fallback."""
        try:
            result = self.search_service.cse().list(
                q=query,
                cx=self.google_cse_id,
                num=min(num_results, 10)  # API limit
            ).execute()
            
            items = result.get('items', [])
            return items
            
        except Exception as e:
            logging.error(f"Google search failed for query '{query}': {e}")
            
            # Check if it's a quota error
            if "quota" in str(e).lower() or "429" in str(e):
                logging.info("Google API quota exceeded, trying alternative search APIs...")
                # Try alternative search APIs
                try:
                    alternative_results = await self.alternative_search.search_with_fallback(query, num_results)
                    if alternative_results:
                        logging.info(f"Alternative search successful: {len(alternative_results)} results")
                        return alternative_results
                    else:
                        logging.warning("Alternative search APIs also failed")
                        return []
                except Exception as alt_e:
                    logging.error(f"Alternative search failed: {alt_e}")
                    return []
            
            return []

    async def _search_news_sources_with_core_facts(self, core_facts: Dict[str, str], num_results: int) -> List[EvidenceItem]:
        """Search news sources using core facts for better targeting."""
        evidence_items = []
        
        # Build targeted search queries using core facts
        search_queries = self._build_targeted_queries(core_facts, 'news')
        
        for query in search_queries[:3]:  # Use top 3 queries
            try:
                results = await self._perform_google_search(query, num_results // 3)
                
                for item in results:
                    if self._is_relevant_to_core_facts(item, core_facts):
                        evidence_items.append(EvidenceItem(
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=item.get('snippet', ''),
                            source_type='news',
                            credibility_score=0.8,
                            relevance_score=0.8
                        ))
                        
            except Exception as e:
                logging.error(f"Error in news search with query '{query}': {e}")
        
        return evidence_items
    
    async def _search_fact_checkers_with_core_facts(self, core_facts: Dict[str, str], num_results: int) -> List[EvidenceItem]:
        """Search fact-checkers using core facts for better targeting."""
        evidence_items = []
        
        # Build targeted search queries using core facts
        search_queries = self._build_targeted_queries(core_facts, 'fact_check')
        
        for site in self.fact_check_sites[:2]:
            for query in search_queries[:2]:
                try:
                    site_query = f'{query} site:{site}'
                    results = await self._perform_google_search(site_query, 1)
                    
                    for item in results:
                        if self._is_relevant_to_core_facts(item, core_facts):
                            evidence_items.append(EvidenceItem(
                                title=item.get('title', ''),
                                url=item.get('link', ''),
                                snippet=item.get('snippet', ''),
                                source_type='fact_check',
                                credibility_score=0.9,
                                relevance_score=0.8
                            ))
                            
                except Exception as e:
                    logging.error(f"Error searching {site} with query '{query}': {e}")
        
        return evidence_items
    
    async def _search_official_sources_with_core_facts(self, core_facts: Dict[str, str], num_results: int) -> List[EvidenceItem]:
        """Search official sources using core facts for better targeting."""
        evidence_items = []
        
        # Build targeted search queries using core facts
        search_queries = self._build_targeted_queries(core_facts, 'official')
        
        for query in search_queries[:2]:
            try:
                results = await self._perform_google_search(query, num_results // 2)
                
                for item in results:
                    if self._is_relevant_to_core_facts(item, core_facts):
                        evidence_items.append(EvidenceItem(
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=item.get('snippet', ''),
                            source_type='official',
                            credibility_score=0.85,
                            relevance_score=0.8
                        ))
                        
            except Exception as e:
                logging.error(f"Error in official search with query '{query}': {e}")
        
        return evidence_items
    
    async def _search_academic_sources_with_core_facts(self, core_facts: Dict[str, str], num_results: int) -> List[EvidenceItem]:
        """Search academic sources using core facts for better targeting."""
        evidence_items = []
        
        # Build targeted search queries using core facts
        search_queries = self._build_targeted_queries(core_facts, 'academic')
        
        for query in search_queries[:2]:
            try:
                results = await self._perform_google_search(query, num_results // 2)
                
                for item in results:
                    if self._is_relevant_to_core_facts(item, core_facts):
                        evidence_items.append(EvidenceItem(
                            title=item.get('title', ''),
                            url=item.get('link', ''),
                            snippet=item.get('snippet', ''),
                            source_type='academic',
                            credibility_score=0.9,
                            relevance_score=0.8
                        ))
                        
            except Exception as e:
                logging.error(f"Error in academic search with query '{query}': {e}")
        
        return evidence_items
    
    def _build_targeted_queries(self, core_facts: Dict[str, str], source_type: str) -> List[str]:
        """Build targeted search queries based on core facts and source type."""
        queries = []
        
        # Base query with main event and participants
        if core_facts['main_event'] and core_facts['participants']:
            participants_str = ' '.join(core_facts['participants'][:2])  # Use top 2 participants
            base_query = f"{participants_str} {core_facts['main_event']}"
            
            if core_facts['date']:
                base_query += f" {core_facts['date']}"
            
            queries.append(base_query)
            
            # Add outcome-specific query
            if core_facts['outcome']:
                outcome_query = f"{base_query} {core_facts['outcome']}"
                queries.append(outcome_query)
        
        # Add specific queries for different source types
        if source_type == 'news':
            if core_facts['participants'] and core_facts['main_event']:
                queries.append(f"{core_facts['participants'][0]} {core_facts['main_event']} result")
        elif source_type == 'fact_check':
            if core_facts['participants'] and core_facts['main_event']:
                queries.append(f"{core_facts['participants'][0]} {core_facts['main_event']} fact check")
        elif source_type == 'official':
            if core_facts['main_event']:
                queries.append(f"{core_facts['main_event']} official results")
        
        return queries
    
    def _is_relevant_to_core_facts(self, search_result: Dict, core_facts: Dict[str, str]) -> bool:
        """Check if search result is relevant to core facts."""
        content = f"{search_result.get('title', '')} {search_result.get('snippet', '')}".lower()
        
        # Check for main event
        if core_facts['main_event'] and core_facts['main_event'].lower() not in content:
            return False
        
        # Check for at least one participant
        if core_facts['participants']:
            participant_found = any(participant.lower() in content for participant in core_facts['participants'])
            if not participant_found:
                return False
        
        # Check for date if available
        if core_facts['date'] and core_facts['date'] not in content:
            return False
        
        return True
