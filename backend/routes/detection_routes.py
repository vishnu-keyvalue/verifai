from flask import Blueprint, request, jsonify
from modules.ai_content_detection import analyze_image_with_model

# Create a Blueprint object. 
# The first argument, 'detection', is the Blueprint's name.
# The second argument, __name__, is standard and helps Flask locate resources.
detection_bp = Blueprint('detection', __name__)

@detection_bp.route("/detect/image", methods=["POST"])
def detect_image_artifacts():
    """
    Endpoint to receive an image file and return an AI content analysis.
    This route is now part of the 'detection' Blueprint.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in the request."}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if image_file:
        analysis_result = analyze_image_with_model(image_file.stream)
        
        if "error" in analysis_result:
            return jsonify(analysis_result), 500
            
        return jsonify(analysis_result), 200
    
    return jsonify({"error": "An unexpected error occurred."}), 500