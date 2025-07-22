import numpy as np
from PIL import Image
import io

# Import TensorFlow and the Keras API
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# --- Model Loading ---
# We load the model once when the application starts.
# This is crucial for performance as model loading is slow.
# The model is loaded into memory and reused for every request.
print("Loading ResNet50 model...")
try:
    # `weights='imagenet'` will automatically download the pre-trained weights on first run.
    model = ResNet50(weights='imagenet')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def analyze_image_with_model(image_stream):
    """
    Analyzes an image using a pre-trained ResNet50 model to find anomalies.

    Args:
        image_stream: A file-like object (stream) containing the image data.

    Returns:
        A dictionary containing the analysis results.
    """
    if model is None:
        return {"error": "Model is not loaded. Cannot perform analysis."}

    try:
        # 1. Image Preprocessing
        # ---------------------
        # The model expects images in a very specific format.
        
        # Load the image from the stream
        image = Image.open(image_stream)

        # Ensure image is in RGB format (some PNGs have an alpha channel)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize the image to 224x224 pixels, as required by ResNet50
        image = image.resize((224, 224))

        # Convert the PIL image to a NumPy array
        image_array = tf.keras.preprocessing.image.img_to_array(image)

        # Expand dimensions to create a "batch" of 1 image
        # The model expects input shape: (batch_size, height, width, channels)
        image_batch = np.expand_dims(image_array, axis=0)

        # Use the special Keras function to preprocess the image (scales pixel values)
        processed_image = preprocess_input(image_batch)

        # 2. Make Prediction (Feature Extraction)
        # ----------------------------------------
        predictions = model.predict(processed_image)

        # Decode the predictions into a human-readable format
        # This gives us the top predicted classes for the image based on ImageNet
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # 3. Apply Heuristic (Our AI Detection Logic)
        # ---------------------------------------------
        # This is a simple heuristic: Real photos usually have one very confident
        # prediction. Some AI images might confuse the model, resulting in lower
        # confidence scores across the top predictions.
        
        top_prediction = decoded_predictions[0]
        _, top_class_name, top_confidence = top_prediction
        
        # We'll set a confidence threshold.
        # If the model's top guess is below this threshold, we flag it as suspicious.
        confidence_threshold = 0.50 

        if top_confidence < confidence_threshold:
            is_likely_ai = True
            reason = f"Model was not confident in its top prediction ({top_class_name}). This can be indicative of unusual or synthetic image patterns."
        else:
            is_likely_ai = False
            reason = "Image patterns align with typical real-world photos recognized by the model."

        # Prepare the response
        result = {
            "is_likely_ai": is_likely_ai,
            "confidence": float(top_confidence), # Convert from NumPy float
            "details": {
                "reason": reason,
                "model_top_prediction": {
                    "class_name": top_class_name,
                    "confidence_score": float(top_confidence)
                },
                "all_predictions": [
                    {"class": pred[1], "score": float(pred[2])} for pred in decoded_predictions
                ]
            }
        }
        return result

    except Exception as e:
        return {"error": f"Failed during model analysis: {str(e)}"}