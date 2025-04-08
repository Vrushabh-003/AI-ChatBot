from flask import Flask, request, jsonify
from flask_cors import CORS
import preprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/get_disease_info", methods=["POST"])
def get_disease_info():
    data = request.json
    user_input = data.get("message")
    
    app.logger.info(f"Received input: '{user_input}'")
    
    if not user_input:
        return jsonify({"error": "Please provide symptoms"}), 400

    try:
        disease = preprocess.find_disease(user_input)
        doctor = preprocess.recommend_doctor(disease)
        precautions = preprocess.get_precaution(disease)
    except Exception as e:
        app.logger.error("Error processing input", exc_info=e)
        return jsonify({"error": "An error occurred during processing."}), 500
    
    response = {
        "predicted_disease": disease,
        "recommended_doctor": doctor,
        "precautions": precautions
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
