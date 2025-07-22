# routes/fact_check_routes.py - Enhanced version
from flask import Blueprint, request, jsonify
import logging
import asyncio
from datetime import datetime

# Import the initialized components from config
import config

# Import data structures
from modules.evidence_aggregator import Evidence

fact_check_bp = Blueprint('fact_check', __name__)
logger = logging.getLogger(__name__)

@fact_check_bp.route("/fact-check/url", methods=["POST"])
def advanced_fact_check_from_url():
    """Enhanced fact-checking endpoint with comprehensive analysis."""
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "Request must include a 'url' key."}), 400
        
        url = data["url"]
        logger.info(f"Starting enhanced fact-check for URL: {url}")
        
        # Step 1: Enhanced Content Extraction
        logger.info("Step 1: Enhanced content extraction...")
        extracted_data = config.content_extractor.extract_comprehensive_content(url)
        if "error" in extracted_data:
            error_msg = extracted_data["error"]
            logger.error(f"Content extraction failed: {error_msg}")
            
            # Provide more specific error messages based on the error type
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                return jsonify({
                    "error": "Unable to access the website. The site may be temporarily unavailable or blocking automated access. Please try again later or check if the URL is correct.",
                    "details": error_msg,
                    "suggestion": "Try using a different news source or check the URL manually in your browser."
                }), 400
            elif "failed after" in error_msg and "attempts" in error_msg:
                return jsonify({
                    "error": "Unable to extract content from this website after multiple attempts. The site may have anti-scraping measures or be temporarily unavailable.",
                    "details": error_msg,
                    "suggestion": "Try using a different news source or paste the article text directly."
                }), 400
            else:
                return jsonify({
                    "error": "Failed to extract content from the provided URL. Please check if the URL is correct and accessible.",
                    "details": error_msg
                }), 400
        
        # Extract claims for analysis
        atomic_claims = extracted_data.get("atomic_claims", [])
        if not atomic_claims:
            return jsonify({"error": "Could not extract verifiable claims from content."}), 400
        
        # Analyze primary claim (most important)
        primary_claim = atomic_claims[0]
        logger.info(f"Analyzing claim: {primary_claim[:100]}...")
        
        # Step 2: Advanced Evidence Retrieval
        logger.info("Step 2: Retrieving comprehensive evidence...")
        evidence_results = asyncio.run(
            config.evidence_retriever.comprehensive_evidence_search(primary_claim, max_results=20)
        )
        
        # Check if quota was exceeded
        quota_exceeded = evidence_results.get('quota_exceeded', False)
        
        # Step 3: Source Credibility Assessment
        logger.info("Step 3: Assessing source credibility...")
        all_evidence_items = []
        source_analyses = []
        
        for source_type, evidence_list in evidence_results.items():
            if source_type == 'quota_exceeded':
                continue  # Skip the quota indicator
                
            for evidence_item in evidence_list:
                # Assess source credibility
                credibility_analysis = config.credibility_assessor.assess_source_credibility(
                    evidence_item.url, evidence_item.snippet
                )
                source_analyses.append(credibility_analysis)
                
                # Create Evidence object for aggregation
                evidence_obj = Evidence(
                    id=f"{source_type}_{len(all_evidence_items)}",
                    content=evidence_item.snippet,
                    source=evidence_item.url,
                    credibility_score=credibility_analysis['credibility_score'],
                    relevance_score=config.semantic_analyzer.calculate_semantic_similarity(
                        primary_claim, evidence_item.snippet
                    ),
                    source_type=source_type
                )
                all_evidence_items.append(evidence_obj)
        
        logger.info(f"Collected {len(all_evidence_items)} evidence pieces")
        
        # Step 4: GEAR-Inspired Evidence Aggregation
        logger.info("Step 4: Advanced evidence aggregation...")
        evidence_assessment = config.evidence_aggregator.aggregate_evidence(
            primary_claim, all_evidence_items
        )
        
        # Step 5: Sophisticated Verdict Generation
        logger.info("Step 5: Generating enhanced verdict...")
        verdict_result = config.verdict_generator.generate_nuanced_verdict(
            primary_claim, evidence_assessment, source_analyses
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Enhanced response format
        final_response = {
            "source_content": {
                "title": extracted_data.get("title"),
                "url": url,
                "domain_info": extracted_data.get("domain_info"),
                "content_quality": extracted_data.get("content_quality_metrics"),
                "atomic_claims": atomic_claims[:5]  # Top 5 claims
            },
            "verdict_analysis": {
                "primary_claim": primary_claim,
                "verdict": verdict_result.verdict,
                "confidence_score": round(verdict_result.confidence_score, 3),
                "explanation": verdict_result.explanation,
                "evidence_summary": verdict_result.evidence_summary,
                "risk_factors": verdict_result.risk_factors,
                "verification_suggestions": verdict_result.verification_suggestions,
                "credibility_breakdown": verdict_result.credibility_breakdown
            },
            "evidence_analysis": {
                "total_sources": len(all_evidence_items),
                "sources_by_type": {
                    source_type: len(evidence_list) 
                    for source_type, evidence_list in evidence_results.items()
                    if source_type != 'quota_exceeded'
                },
                "average_credibility": round(
                    sum(s['credibility_score'] for s in source_analyses) / len(source_analyses)
                    if source_analyses else 0, 3
                ),
                "api_quota_exceeded": quota_exceeded
            },
            "processing_metadata": {
                "processing_time_seconds": round(processing_time, 2),
                "timestamp": datetime.now().isoformat(),
                "version": "2.0-enhanced"
            }
        }
        
        # Add quota exceeded warning if applicable
        if quota_exceeded:
            final_response["verdict_analysis"]["explanation"] += " | Note: Search API quota exceeded - limited evidence available"
            final_response["verdict_analysis"]["risk_factors"].append("Search API quota exceeded - evidence may be incomplete")
        
        logger.info(f"Enhanced fact-check completed in {processing_time:.2f}s")
        return jsonify(final_response), 200
        
    except Exception as e:
        logger.error(f"Error in enhanced fact-check: {e}", exc_info=True)
        return jsonify({
            "error": "An error occurred during enhanced fact-checking.",
            "details": str(e)
        }), 500

@fact_check_bp.route("/fact-check/text", methods=["POST"])
def advanced_fact_check_from_text():
    """Enhanced text-based fact-checking."""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Request must include a 'text' key."}), 400
        
        claim = data["text"].strip()
        if len(claim) < 10:
            return jsonify({"error": "Text must be at least 10 characters long."}), 400
        
        logger.info(f"Analyzing text claim: {claim[:100]}...")
        
        # Direct evidence retrieval for text
        logger.info("Starting evidence retrieval...")
        evidence_results = asyncio.run(
            config.evidence_retriever.comprehensive_evidence_search(claim, max_results=15)
        )
        
        logger.info(f"Evidence retrieval completed. Results: {evidence_results}")
        
        # Process evidence same as URL endpoint
        all_evidence_items = []
        source_analyses = []
        
        for source_type, evidence_list in evidence_results.items():
            if source_type == 'quota_exceeded':
                continue  # Skip the quota indicator
                
            # Check if evidence_list is actually a list and not an exception
            if not isinstance(evidence_list, list):
                logger.warning(f"Skipping {source_type} search results - not a list: {type(evidence_list)}")
                continue
                
            logger.info(f"Processing {source_type} evidence: {len(evidence_list)} items")
            
            for evidence_item in evidence_list:
                credibility_analysis = config.credibility_assessor.assess_source_credibility(
                    evidence_item.url, evidence_item.snippet
                )
                source_analyses.append(credibility_analysis)
                
                evidence_obj = Evidence(
                    id=f"{source_type}_{len(all_evidence_items)}",
                    content=evidence_item.snippet,
                    source=evidence_item.url,
                    credibility_score=credibility_analysis['credibility_score'],
                    relevance_score=config.semantic_analyzer.calculate_semantic_similarity(
                        claim, evidence_item.snippet
                    ),
                    source_type=source_type
                )
                all_evidence_items.append(evidence_obj)
        
        logger.info(f"Total evidence items collected: {len(all_evidence_items)}")
        
        evidence_assessment = config.evidence_aggregator.aggregate_evidence(claim, all_evidence_items)
        
        # Check if we have any evidence to work with
        if not all_evidence_items:
            return jsonify({
                "claim": claim,
                "verdict": "Insufficient Evidence",
                "confidence_score": 0.0,
                "explanation": "Unable to find sufficient evidence to verify this claim. This could be due to the claim being too recent, too specific, or not widely reported.",
                "evidence_summary": "No evidence sources were found to verify this claim.",
                "risk_factors": ["Limited evidence available", "Claim may be too recent or too specific"],
                "verification_suggestions": [
                    "Try rephrasing the claim with more specific details",
                    "Check if the claim is from a recent event that may not have been fact-checked yet",
                    "Consider searching for the claim manually on fact-checking websites"
                ],
                "processing_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.0-enhanced"
                }
            }), 200
        
        verdict_result = config.verdict_generator.generate_nuanced_verdict(
            claim, evidence_assessment, source_analyses
        )
        
        return jsonify({
            "claim": claim,
            "verdict": verdict_result.verdict,
            "confidence_score": round(verdict_result.confidence_score, 3),
            "explanation": verdict_result.explanation,
            "evidence_summary": verdict_result.evidence_summary,
            "risk_factors": verdict_result.risk_factors,
            "verification_suggestions": verdict_result.verification_suggestions,
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "2.0-enhanced"
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in text fact-check: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@fact_check_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify backend components are working."""
    try:
        # Test basic component availability
        components_status = {
            "content_extractor": "available" if config.content_extractor else "error",
            "evidence_retriever": "available" if config.evidence_retriever else "error",
            "semantic_analyzer": "available" if config.semantic_analyzer else "error",
            "credibility_assessor": "available" if config.credibility_assessor else "error",
            "evidence_aggregator": "available" if config.evidence_aggregator else "error",
            "verdict_generator": "available" if config.verdict_generator else "error"
        }
        
        return jsonify({
            "status": "healthy",
            "components": components_status,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
