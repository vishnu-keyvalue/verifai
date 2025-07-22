import numpy as np
from PIL import Image
import io
import cv2
from scipy import ndimage
from scipy.fft import fft2, fftshift
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
# We'll use a more sophisticated approach with multiple detection methods
print("Initializing AI Content Detection System...")

# Try to load a pre-trained model for AI detection
try:
    # Using a model that's been fine-tuned for AI-generated image detection
    # If this specific model isn't available, we'll fall back to our custom analysis
    model_name = "microsoft/resnet-50"  # We'll use this as base and apply our own logic
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    print("Base model loaded successfully for feature extraction.")
    base_model_available = True
except Exception as e:
    print(f"Could not load pre-trained model: {e}")
    print("Falling back to custom analysis methods.")
    base_model_available = False
    model = None
    processor = None

def analyze_image_with_model(image_stream):
    """
    Analyzes an image using multiple techniques to detect AI-generated content.

    Args:
        image_stream: A file-like object (stream) containing the image data.

    Returns:
        A dictionary containing the analysis results.
    """
    try:
        # Load the image from the stream
        image = Image.open(image_stream)
        
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert PIL image to numpy array for analysis
        image_array = np.array(image)
        
        logger.info(f"Image loaded: shape={image_array.shape}, dtype={image_array.dtype}")
        
        # Perform multiple analysis techniques
        results = {}
        
        # 1. Frequency Domain Analysis
        logger.info("Starting frequency domain analysis...")
        freq_analysis = analyze_frequency_domain(image_array)
        results['frequency_analysis'] = freq_analysis
        logger.info(f"Frequency analysis result: {freq_analysis}")
        
        # 2. Statistical Analysis
        logger.info("Starting statistical analysis...")
        stats_analysis = analyze_statistical_properties(image_array)
        results['statistical_analysis'] = stats_analysis
        logger.info(f"Statistical analysis result: {stats_analysis}")
        
        # 3. Metadata and Artifact Analysis
        logger.info("Starting artifact analysis...")
        artifact_analysis = analyze_artifacts(image_array)
        results['artifact_analysis'] = artifact_analysis
        logger.info(f"Artifact analysis result: {artifact_analysis}")
        
        # 4. Deep Learning Analysis (if model is available)
        if base_model_available and model is not None:
            logger.info("Starting deep learning analysis...")
            dl_analysis = analyze_with_deep_learning(image)
            results['deep_learning_analysis'] = dl_analysis
            logger.info(f"Deep learning analysis result: {dl_analysis}")
        else:
            logger.info("Deep learning analysis not available")
            results['deep_learning_analysis'] = {'available': False, 'score': 0.0}
        
        # 5. Combine all analyses for final decision
        logger.info("Combining analyses...")
        final_result = combine_analyses(results)
        logger.info(f"Final result: {final_result}")
        
        return final_result

    except Exception as e:
        logger.error(f"Error during image analysis: {e}")
        return {"error": f"Failed during image analysis: {str(e)}"}

def analyze_frequency_domain(image_array):
    """
    Analyze image in frequency domain to detect AI-generated patterns.
    AI-generated images often have different frequency characteristics.
    """
    try:
        logger.info(f"Frequency analysis: input shape={image_array.shape}")
        
        # Convert to grayscale for frequency analysis
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        logger.info(f"Frequency analysis: gray shape={gray.shape}, range=[{gray.min()}, {gray.max()}]")
        
        # Apply FFT
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        logger.info(f"Frequency analysis: magnitude spectrum shape={magnitude_spectrum.shape}")
        
        # Analyze frequency distribution
        # AI-generated images often have more uniform frequency distributions
        freq_std = np.std(magnitude_spectrum)
        freq_mean = np.mean(magnitude_spectrum)
        freq_entropy = calculate_entropy(magnitude_spectrum)
        
        logger.info(f"Frequency analysis: freq_std={freq_std:.3f}, freq_mean={freq_mean:.3f}, freq_entropy={freq_entropy:.3f}")
        
        # Check for unusual frequency patterns
        # Real images typically have more varied frequency content
        frequency_score = 0.0
        frequency_reasons = []
        
        if freq_entropy < 2.0:  # Very strict threshold for low entropy
            frequency_score += 0.5
            frequency_reasons.append("Extremely low frequency entropy suggests artificial patterns")
            logger.info("Frequency analysis: Extremely low entropy detected")
        elif freq_entropy < 3.0:  # Strict threshold
            frequency_score += 0.3
            frequency_reasons.append("Very low frequency entropy suggests artificial patterns")
            logger.info("Frequency analysis: Very low entropy detected")
        elif freq_entropy < 4.0:  # Moderate threshold
            frequency_score += 0.1
            frequency_reasons.append("Low frequency entropy suggests artificial patterns")
            logger.info("Frequency analysis: Low entropy detected")
            
        if freq_std < 0.8:  # Very strict threshold for low std
            frequency_score += 0.4
            frequency_reasons.append("Very uniform frequency distribution")
            logger.info("Frequency analysis: Very low std detected")
        elif freq_std < 1.5:  # Strict threshold
            frequency_score += 0.2
            frequency_reasons.append("Uniform frequency distribution")
            logger.info("Frequency analysis: Low std detected")
            
        # Check for grid-like patterns (common in AI-generated images)
        grid_score = detect_grid_patterns(magnitude_spectrum)
        logger.info(f"Frequency analysis: grid_score={grid_score:.3f}")
        if grid_score > 0.8:  # Higher threshold for grid patterns
            frequency_score += 0.3
            frequency_reasons.append("Strong grid-like frequency patterns detected")
            logger.info("Frequency analysis: Strong grid patterns detected")
        elif grid_score > 0.6:
            frequency_score += 0.1
            frequency_reasons.append("Grid-like frequency patterns detected")
            logger.info("Frequency analysis: Grid patterns detected")
        
        final_score = min(frequency_score, 1.0)
        logger.info(f"Frequency analysis: final_score={final_score:.3f}, reasons={frequency_reasons}")
        
        return {
            'score': final_score,
            'reasons': frequency_reasons,
            'metrics': {
                'freq_std': float(freq_std),
                'freq_mean': float(freq_mean),
                'freq_entropy': float(freq_entropy),
                'grid_score': float(grid_score)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in frequency analysis: {e}")
        return {'score': 0.0, 'reasons': ['Frequency analysis failed'], 'metrics': {}}

def analyze_statistical_properties(image_array):
    """
    Analyze statistical properties of the image.
    AI-generated images often have different statistical characteristics.
    """
    try:
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Calculate various statistical measures
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        skewness = calculate_skewness(gray)
        kurtosis = calculate_kurtosis(gray)
        
        # Calculate local variance (texture measure)
        local_variance = calculate_local_variance(gray)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Analyze color distribution if color image
        color_analysis = {}
        if len(image_array.shape) == 3:
            color_analysis = analyze_color_distribution(image_array)
        
        # Score based on statistical properties
        stats_score = 0.0
        stats_reasons = []
        
        # Check for unusual intensity distributions
        if std_intensity < 20:  # Very low variance suggests artificial image
            stats_score += 0.3
            stats_reasons.append("Very low intensity variance")
        elif std_intensity < 35:  # Low variance
            stats_score += 0.15
            stats_reasons.append("Low intensity variance")
            
        if abs(skewness) < 0.05:  # Very symmetric distribution
            stats_score += 0.25
            stats_reasons.append("Unusually symmetric intensity distribution")
        elif abs(skewness) < 0.15:  # Somewhat symmetric
            stats_score += 0.1
            stats_reasons.append("Symmetric intensity distribution")
            
        if local_variance < 50:  # Very low texture
            stats_score += 0.3
            stats_reasons.append("Very low texture variation")
        elif local_variance < 150:  # Low texture
            stats_score += 0.15
            stats_reasons.append("Low texture variation")
            
        if edge_density < 0.005:  # Very few edges
            stats_score += 0.25
            stats_reasons.append("Very low edge density")
        elif edge_density < 0.015:  # Low edge density
            stats_score += 0.1
            stats_reasons.append("Low edge density")
            
        # Color analysis
        if color_analysis.get('saturation_score', 0) > 0.5:
            stats_score += 0.1
            stats_reasons.append("Unusual color saturation patterns")
        
        return {
            'score': min(stats_score, 1.0),
            'reasons': stats_reasons,
            'metrics': {
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'local_variance': float(local_variance),
                'edge_density': float(edge_density),
                **color_analysis
            }
        }
        
    except Exception as e:
        logger.error(f"Error in statistical analysis: {e}")
        return {'score': 0.0, 'reasons': ['Statistical analysis failed'], 'metrics': {}}

def analyze_artifacts(image_array):
    """
    Look for common artifacts in AI-generated images.
    """
    try:
        artifacts_score = 0.0
        artifact_reasons = []
        
        # Check for compression artifacts
        compression_score = detect_compression_artifacts(image_array)
        if compression_score > 0.3:
            artifacts_score += 0.2
            artifact_reasons.append("Compression artifacts detected")
        
        # Check for repeating patterns
        pattern_score = detect_repeating_patterns(image_array)
        if pattern_score > 0.4:
            artifacts_score += 0.3
            artifact_reasons.append("Repeating patterns detected")
        
        # Check for unrealistic textures
        texture_score = analyze_texture_realism(image_array)
        if texture_score > 0.5:
            artifacts_score += 0.25
            artifact_reasons.append("Unrealistic texture patterns")
        
        # Check for geometric inconsistencies
        geometry_score = detect_geometric_inconsistencies(image_array)
        if geometry_score > 0.4:
            artifacts_score += 0.25
            artifact_reasons.append("Geometric inconsistencies detected")
        
        return {
            'score': min(artifacts_score, 1.0),
            'reasons': artifact_reasons,
            'metrics': {
                'compression_score': float(compression_score),
                'pattern_score': float(pattern_score),
                'texture_score': float(texture_score),
                'geometry_score': float(geometry_score)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in artifact analysis: {e}")
        return {'score': 0.0, 'reasons': ['Artifact analysis failed'], 'metrics': {}}

def analyze_with_deep_learning(image):
    """
    Use pre-trained model for feature extraction and analysis.
    """
    try:
        # Preprocess image for the model
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.logits
            
        # Extract features and analyze
        feature_vector = features.numpy().flatten()
        
        # Simple analysis based on feature statistics
        feature_mean = np.mean(feature_vector)
        feature_std = np.std(feature_vector)
        feature_entropy = calculate_entropy(feature_vector)
        
        # Score based on feature characteristics
        dl_score = 0.0
        dl_reasons = []
        
        if feature_std < 1.0:  # Low feature variance
            dl_score += 0.3
            dl_reasons.append("Low feature variance in deep learning analysis")
            
        if feature_entropy < 5.0:  # Low feature entropy
            dl_score += 0.2
            dl_reasons.append("Low feature entropy suggests artificial patterns")
        
        return {
            'score': min(dl_score, 1.0),
            'reasons': dl_reasons,
            'metrics': {
                'feature_mean': float(feature_mean),
                'feature_std': float(feature_std),
                'feature_entropy': float(feature_entropy)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in deep learning analysis: {e}")
        return {'score': 0.0, 'reasons': ['Deep learning analysis failed'], 'metrics': {}}

def combine_analyses(results):
    """
    Combine all analysis results to make final decision.
    """
    try:
        # Weight different analyses
        weights = {
            'frequency_analysis': 0.3,
            'statistical_analysis': 0.25,
            'artifact_analysis': 0.3,
            'deep_learning_analysis': 0.15
        }
        
        total_score = 0.0
        all_reasons = []
        confidence = 0.0
        
        for analysis_name, weight in weights.items():
            if analysis_name in results:
                analysis = results[analysis_name]
                if 'score' in analysis:
                    total_score += analysis['score'] * weight
                    all_reasons.extend(analysis.get('reasons', []))
                    confidence += weight
        
        # Normalize score
        if confidence > 0:
            final_score = total_score / confidence
        else:
            final_score = 0.0
        
        # Determine if likely AI-generated - lower threshold for better sensitivity
        is_likely_ai = final_score > 0.4
        
        # Generate explanation
        if is_likely_ai:
            reason = f"Multiple analysis techniques detected patterns consistent with AI-generated content (AI likelihood: {final_score:.2f})."
        else:
            reason = f"Analysis suggests this is likely a real image (AI likelihood: {final_score:.2f})."
        
        # Add specific reasons if available
        if all_reasons:
            reason += f" Key indicators: {', '.join(all_reasons[:3])}."
        
        return {
            "is_likely_ai": is_likely_ai,
            "confidence": float(final_score),
            "details": {
                "reason": reason,
                "analysis_method": "Multi-technique AI detection",
                "techniques_used": list(results.keys()),
                "detailed_scores": {
                    name: result.get('score', 0.0) 
                    for name, result in results.items() 
                    if 'score' in result
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error combining analyses: {e}")
        return {"error": f"Failed to combine analysis results: {str(e)}"}

# Helper functions
def calculate_entropy(array):
    """Calculate entropy of an array."""
    try:
        hist, _ = np.histogram(array, bins=256, range=(0, 256))
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob))
    except:
        return 0.0

def calculate_skewness(array):
    """Calculate skewness of an array."""
    try:
        mean = np.mean(array)
        std = np.std(array)
        if std == 0:
            return 0.0
        return np.mean(((array - mean) / std) ** 3)
    except:
        return 0.0

def calculate_kurtosis(array):
    """Calculate kurtosis of an array."""
    try:
        mean = np.mean(array)
        std = np.std(array)
        if std == 0:
            return 0.0
        return np.mean(((array - mean) / std) ** 4) - 3
    except:
        return 0.0

def calculate_local_variance(image, window_size=5):
    """Calculate local variance as a texture measure."""
    try:
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        mean_img = ndimage.convolve(image.astype(float), kernel)
        var_img = ndimage.convolve((image.astype(float) ** 2), kernel) - (mean_img ** 2)
        return np.mean(var_img)
    except:
        return 0.0

def detect_grid_patterns(magnitude_spectrum):
    """Detect grid-like patterns in frequency domain."""
    try:
        # Look for regular patterns in the magnitude spectrum
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Check for regular spacing in frequency domain
        horizontal_profile = np.mean(magnitude_spectrum, axis=0)
        vertical_profile = np.mean(magnitude_spectrum, axis=1)
        
        # Calculate autocorrelation to detect periodicity
        h_autocorr = np.correlate(horizontal_profile, horizontal_profile, mode='full')
        v_autocorr = np.correlate(vertical_profile, vertical_profile, mode='full')
        
        # Normalize autocorrelation
        h_autocorr = h_autocorr / h_autocorr.max()
        v_autocorr = v_autocorr / v_autocorr.max()
        
        # Look for peaks in autocorrelation (indicating periodicity)
        h_peaks = np.sum(h_autocorr > 0.8)
        v_peaks = np.sum(v_autocorr > 0.8)
        
        # Score based on number of peaks
        grid_score = (h_peaks + v_peaks) / (h + w) * 10
        return min(grid_score, 1.0)
    except:
        return 0.0

def analyze_color_distribution(image_array):
    """Analyze color distribution characteristics."""
    try:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Analyze saturation
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # Analyze hue distribution
        hue = hsv[:, :, 0]
        hue_entropy = calculate_entropy(hue)
        
        # Score based on color characteristics
        saturation_score = 0.0
        if sat_std < 20:  # Very uniform saturation
            saturation_score = 0.5
        elif sat_std > 80:  # Very varied saturation
            saturation_score = 0.3
            
        return {
            'saturation_score': saturation_score,
            'sat_mean': float(sat_mean),
            'sat_std': float(sat_std),
            'hue_entropy': float(hue_entropy)
        }
    except:
        return {'saturation_score': 0.0}

def detect_compression_artifacts(image_array):
    """Detect compression artifacts."""
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Apply high-pass filter to detect artifacts
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        
        # Calculate artifact score
        artifact_intensity = np.mean(np.abs(filtered))
        return min(artifact_intensity / 50.0, 1.0)
    except:
        return 0.0

def detect_repeating_patterns(image_array):
    """Detect repeating patterns in the image."""
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Use autocorrelation to detect repeating patterns
        h, w = gray.shape
        autocorr = np.correlate2d(gray, gray, mode='full')
        
        # Normalize
        autocorr = autocorr / autocorr.max()
        
        # Look for regular peaks in autocorrelation
        peaks = np.sum(autocorr > 0.8)
        pattern_score = peaks / (h * w) * 100
        
        return min(pattern_score, 1.0)
    except:
        return 0.0

def analyze_texture_realism(image_array):
    """Analyze texture realism."""
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Calculate texture measures
        # 1. Local binary patterns
        lbp = calculate_lbp(gray)
        lbp_entropy = calculate_entropy(lbp)
        
        # 2. Gray-level co-occurrence matrix features
        glcm_features = calculate_glcm_features(gray)
        
        # Score based on texture characteristics
        texture_score = 0.0
        
        if lbp_entropy < 4.0:  # Low texture variety
            texture_score += 0.3
            
        if glcm_features['contrast'] < 50:  # Low contrast
            texture_score += 0.2
            
        if glcm_features['homogeneity'] > 0.8:  # Too homogeneous
            texture_score += 0.2
            
        return texture_score
    except:
        return 0.0

def calculate_lbp(image, radius=1, n_points=8):
    """Calculate Local Binary Pattern."""
    try:
        # Simple LBP implementation
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if image[x, y] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
                
        return lbp
    except:
        return np.zeros_like(image)

def calculate_glcm_features(image):
    """Calculate Gray-Level Co-occurrence Matrix features."""
    try:
        # Simplified GLCM calculation
        h, w = image.shape
        glcm = np.zeros((256, 256))
        
        # Calculate GLCM for horizontal direction
        for i in range(h):
            for j in range(w - 1):
                glcm[image[i, j], image[i, j + 1]] += 1
                
        # Normalize
        glcm = glcm / glcm.sum()
        
        # Calculate features
        contrast = np.sum(glcm * np.square(np.arange(256)[:, None] - np.arange(256)))
        homogeneity = np.sum(glcm / (1 + np.square(np.arange(256)[:, None] - np.arange(256))))
        
        return {
            'contrast': float(contrast),
            'homogeneity': float(homogeneity)
        }
    except:
        return {'contrast': 0.0, 'homogeneity': 0.0}

def detect_geometric_inconsistencies(image_array):
    """Detect geometric inconsistencies."""
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour properties
        geometry_score = 0.0
        
        if len(contours) > 0:
            # Check for unrealistic contour properties
            areas = [cv2.contourArea(c) for c in contours]
            perimeters = [cv2.arcLength(c, True) for c in contours]
            
            # Calculate circularity
            circularities = [4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0 
                           for area, perimeter in zip(areas, perimeters)]
            
            # Check for too many perfect circles (unrealistic)
            perfect_circles = sum(1 for c in circularities if c > 0.9)
            if perfect_circles > len(contours) * 0.3:
                geometry_score += 0.3
                
            # Check for unrealistic area distributions
            if len(areas) > 1:
                area_std = np.std(areas)
                if area_std < 100:  # Very uniform areas
                    geometry_score += 0.2
                    
        return geometry_score
    except:
        return 0.0