# modules/source_credibility_assessor.py
import requests
import whois
from datetime import datetime
from urllib.parse import urlparse
import ssl
import socket
from typing import Dict, List, Optional
import logging
import re

class SourceCredibilityAssessor:
    def __init__(self):
        
        # Known high-credibility domains
        self.high_credibility_domains = {
            # Fact-checking sites
            'snopes.com': 0.95,
            'factcheck.org': 0.95,
            'politifact.com': 0.90,
            'reuters.com': 0.90,
            'apnews.com': 0.90,
            
            # Academic institutions
            'harvard.edu': 0.95,
            'mit.edu': 0.95,
            'stanford.edu': 0.95,
            
            # Government sources
            'cdc.gov': 0.95,
            'nih.gov': 0.95,
            'nasa.gov': 0.95,
            'who.int': 0.95,
            
            # Reputable news
            'bbc.com': 0.85,
            'npr.org': 0.85,
            'theguardian.com': 0.80,
            'wsj.com': 0.80,
            'nytimes.com': 0.80
        }
        
        # Known problematic domains
        self.low_credibility_domains = {
            'infowars.com': 0.1,
            'naturalnews.com': 0.2,
            'breitbart.com': 0.3,
            'dailymail.co.uk': 0.4
        }
    
    def assess_source_credibility(self, url: str, content: str = "") -> Dict:
        """Comprehensive source credibility assessment."""
        domain = self._extract_domain(url)
        
        credibility_factors = {
            'known_source_rating': self._get_known_rating(domain),
            'domain_age_score': self._assess_domain_age(domain),
            'security_score': self._assess_security_features(url),
            'transparency_score': self._assess_transparency(content, url),
            'content_quality_score': self._assess_content_quality_indicators(content),
            'technical_indicators_score': self._assess_technical_indicators(url)
        }
        
        # Calculate composite credibility score
        composite_score = self._calculate_weighted_credibility(credibility_factors)
        
        # Generate risk assessment
        risk_level = self._categorize_risk_level(composite_score)
        
        return {
            'domain': domain,
            'credibility_score': composite_score,
            'credibility_factors': credibility_factors,
            'risk_level': risk_level,
            'verification_notes': self._generate_verification_notes(credibility_factors),
            'recommendations': self._generate_recommendations(composite_score)
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL."""
        domain = urlparse(url).netloc.lower()
        return domain.replace('www.', '')
    
    def _get_known_rating(self, domain: str) -> float:
        """Get rating for known domains."""
        if domain in self.high_credibility_domains:
            return self.high_credibility_domains[domain]
        elif domain in self.low_credibility_domains:
            return self.low_credibility_domains[domain]
        else:
            return 0.5  # Neutral for unknown domains
    
    def _assess_domain_age(self, domain: str) -> float:
        """Assess credibility based on domain age."""
        try:
            whois_info = whois.whois(domain)
            if whois_info.creation_date:
                creation_date = whois_info.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                
                age = datetime.now() - creation_date
                age_years = age.days / 365.25
                
                # Older domains are generally more credible
                if age_years >= 10:
                    return 1.0
                elif age_years >= 5:
                    return 0.8
                elif age_years >= 2:
                    return 0.6
                elif age_years >= 1:
                    return 0.4
                else:
                    return 0.2
        except Exception as e:
            logging.warning(f"Domain age assessment failed for {domain}: {e}")
            return 0.5
        
        return 0.5
    
    def _assess_security_features(self, url: str) -> float:
        """Assess security indicators."""
        score = 0.0
        
        # Check HTTPS
        if url.startswith('https://'):
            score += 0.4
        
        # Check SSL certificate validity
        try:
            domain = urlparse(url).netloc
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    if cert:
                        score += 0.3  # Valid SSL certificate
                        
                        # Check certificate age
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        if not_after > datetime.now():
                            score += 0.3  # Certificate not expired
        except Exception:
            pass  # SSL check failed
        
        return min(score, 1.0)
    
    def _assess_transparency(self, content: str, url: str) -> float:
        """Assess transparency indicators."""
        score = 0.0
        domain = self._extract_domain(url)
        
        # Check for contact information
        contact_indicators = [
            r'contact@', r'info@', r'editor@', r'newsroom@',
            r'phone:', r'tel:', r'address:', r'office:'
        ]
        
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in contact_indicators):
            score += 0.3
        
        # Check for about page (common URLs)
        about_urls = [
            f"https://{domain}/about",
            f"https://{domain}/about-us",
            f"https://www.{domain}/about"
        ]
        
        for about_url in about_urls:
            try:
                response = requests.head(about_url, timeout=5)
                if response.status_code == 200:
                    score += 0.3
                    break
            except Exception:
                continue
        
        # Check for author information
        if re.search(r'\bby\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b', content):
            score += 0.2
        
        # Check for date information
        if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\w+\s+\d{1,2},\s+\d{4}\b', content):
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_content_quality_indicators(self, content: str) -> float:
        """Assess content quality indicators."""
        if not content:
            return 0.5
        
        score = 0.0
        
        # Length indicator (longer articles often more comprehensive)
        word_count = len(content.split())
        if word_count >= 1000:
            score += 0.2
        elif word_count >= 500:
            score += 0.15
        elif word_count >= 200:
            score += 0.1
        
        # Source citations
        citation_patterns = [
            r'according to', r'source:', r'study shows', r'research indicates',
            r'data from', r'statistics show', r'report states'
        ]
        citation_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                           for pattern in citation_patterns)
        if citation_count >= 5:
            score += 0.3
        elif citation_count >= 3:
            score += 0.2
        elif citation_count >= 1:
            score += 0.1
        
        # Quote indicators
        quote_count = content.count('"') + content.count("'")
        if quote_count >= 10:
            score += 0.2
        elif quote_count >= 5:
            score += 0.1
        
        # Balanced reporting (presence of multiple perspectives)
        balance_indicators = ['however', 'although', 'on the other hand', 'critics argue']
        if any(indicator in content.lower() for indicator in balance_indicators):
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_technical_indicators(self, url: str) -> float:
        """Assess technical quality indicators."""
        score = 0.0
        
        try:
            response = requests.head(url, timeout=10)
            
            # Check response status
            if response.status_code == 200:
                score += 0.3
            
            # Check for proper headers
            if 'server' in response.headers:
                score += 0.1
            
            if 'cache-control' in response.headers:
                score += 0.1
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                score += 0.2
            
            # Check for security headers
            security_headers = ['x-frame-options', 'x-content-type-options', 'x-xss-protection']
            security_header_count = sum(1 for header in security_headers 
                                      if header in response.headers)
            score += (security_header_count / len(security_headers)) * 0.3
            
        except Exception as e:
            logging.warning(f"Technical assessment failed for {url}: {e}")
            score = 0.3  # Neutral score if technical check fails
        
        return min(score, 1.0)
    

    
    def _calculate_weighted_credibility(self, factors: Dict) -> float:
        """Calculate weighted composite credibility score."""
        weights = {
            'known_source_rating': 0.25,
            'domain_age_score': 0.15,
            'security_score': 0.15,
            'transparency_score': 0.2,
            'content_quality_score': 0.15,
            'technical_indicators_score': 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if factor in factors and factors[factor] is not None:
                weighted_score += factors[factor] * weight
                total_weight += weight
        
        # Normalize by actual total weight (in case some factors are missing)
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.5  # Neutral score if no factors available
    
    def _categorize_risk_level(self, score: float) -> str:
        """Categorize risk level based on credibility score."""
        if score >= 0.8:
            return "very_low"
        elif score >= 0.6:
            return "low"
        elif score >= 0.4:
            return "moderate"
        elif score >= 0.2:
            return "high"
        else:
            return "very_high"
    
    def _generate_verification_notes(self, factors: Dict) -> List[str]:
        """Generate human-readable verification notes."""
        notes = []
        
        if factors['known_source_rating'] >= 0.8:
            notes.append("Source is from a well-established, credible domain")
        elif factors['known_source_rating'] <= 0.3:
            notes.append("Source domain has credibility concerns")
        
        if factors['security_score'] >= 0.7:
            notes.append("Good security practices (HTTPS, valid certificates)")
        elif factors['security_score'] <= 0.3:
            notes.append("Security concerns detected")
        
        if factors['transparency_score'] >= 0.7:
            notes.append("Good transparency (contact info, author details)")
        elif factors['transparency_score'] <= 0.3:
            notes.append("Limited transparency information")
        
        return notes
    
    def _generate_recommendations(self, score: float) -> List[str]:
        """Generate recommendations based on credibility score."""
        if score >= 0.8:
            return ["Source appears highly credible", "Safe to use as primary evidence"]
        elif score >= 0.6:
            return ["Source appears credible", "Consider cross-referencing with additional sources"]
        elif score >= 0.4:
            return ["Source has mixed credibility", "Verify claims with more reliable sources"]
        elif score >= 0.2:
            return ["Source credibility questionable", "Use with caution and seek verification"]
        else:
            return ["Source has low credibility", "Avoid using as evidence", "Seek alternative sources"]
