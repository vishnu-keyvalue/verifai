# app.py - Updated with config approach
from flask import Flask
from flask_cors import CORS
import logging
import os
from dotenv import load_dotenv
import config  # Import the config module

# Import routes
from routes.detection_routes import detection_bp
from routes.fact_check_routes import fact_check_bp

# Import components
from modules.semantic_analyzer import AdvancedSemanticAnalyzer
from modules.enhanced_content_extractor import EnhancedContentExtractor
from modules.advanced_evidence_retriever import AdvancedEvidenceRetriever
from modules.source_credibility_assessor import SourceCredibilityAssessor
from modules.evidence_aggregator import GEARInspiredEvidenceAggregator
from modules.advanced_verdict_generator import AdvancedVerdictGenerator

# Load environment variables
load_dotenv('../.env')  # Load from parent directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('verifai.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)

def initialize_components():
    """Initialize all enhanced AI components."""
    logging.info("Initializing VerifAI enhanced components...")
    
    # Log API key status (without exposing the actual keys)
    google_api_key = os.getenv('GOOGLE_API_KEY')
    google_cse_id = os.getenv('GOOGLE_CSE_ID')
    
    logging.info(f"Google API Key configured: {'Yes' if google_api_key else 'No'}")
    logging.info(f"Google CSE ID configured: {'Yes' if google_cse_id else 'No'}")
    
    config.semantic_analyzer = AdvancedSemanticAnalyzer(
        model_name=os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
    )
    
    config.content_extractor = EnhancedContentExtractor()
    
    config.evidence_retriever = AdvancedEvidenceRetriever(
        google_api_key=google_api_key,
        google_cse_id=google_cse_id
    )
    
    config.credibility_assessor = SourceCredibilityAssessor()
    
    config.evidence_aggregator = GEARInspiredEvidenceAggregator(config.semantic_analyzer)
    
    config.verdict_generator = AdvancedVerdictGenerator(
        config.evidence_aggregator, config.credibility_assessor
    )
    
    logging.info("All components initialized successfully!")

# Initialize components
initialize_components()

# Register blueprints
app.register_blueprint(detection_bp, url_prefix='/api/v1')
app.register_blueprint(fact_check_bp, url_prefix='/api/v1')

@app.route("/")
def index():
    return "<h1>VerifAI Backend is Running - Enhanced Version 2.0</h1>"

@app.route("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.0-enhanced",
        "components": {
            "semantic_analyzer": "operational" if config.semantic_analyzer else "error",
            "content_extractor": "operational" if config.content_extractor else "error",
            "evidence_retriever": "operational" if config.evidence_retriever else "error",
            "credibility_assessor": "operational" if config.credibility_assessor else "error",
            "verdict_generator": "operational" if config.verdict_generator else "error"
        }
    }

if __name__ == "__main__":
    app.run(debug=True, port=5001)
