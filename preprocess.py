# import pandas as pd
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# # Download NLTK resources if not already downloaded
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("punkt_tab")


# # --------------------------
# # Load and Process Symptoms Data
# # --------------------------
# # Read the DiseaseAndSymptoms CSV file
# df_symptoms = pd.read_csv("DiseaseAndSymptoms.csv")

# # Combine all columns starting with "Symptom_" into a single list for each disease
# symptom_columns = [col for col in df_symptoms.columns if col.startswith("Symptom_")]
# df_symptoms["symptoms"] = df_symptoms[symptom_columns].apply(
#     lambda row: [str(s).strip().lower() for s in row if pd.notna(s)], axis=1
# )

# # Create a dictionary mapping disease name to its list of symptoms
# # Note: The header for disease is "Disease" (capitalized) as per your file
# disease_symptom_dict = dict(zip(df_symptoms["Disease"], df_symptoms["symptoms"]))

# # --------------------------
# # Load and Process Precautions Data
# # --------------------------
# # Read the Disease_precaution CSV file
# df_precaution = pd.read_csv("Disease_precaution.csv")

# # Combine all columns starting with "Precaution_" into a single list for each disease
# precaution_columns = [col for col in df_precaution.columns if col.startswith("Precaution_")]
# df_precaution["precautions"] = df_precaution[precaution_columns].apply(
#     lambda row: [str(p).strip() for p in row if pd.notna(p)], axis=1
# )

# # Create a dictionary mapping disease name to its list of precautions
# disease_precaution_dict = dict(zip(df_precaution["Disease"], df_precaution["precautions"]))

# # --------------------------
# # Helper Functions
# # --------------------------
# def preprocess_text(text):
#     """
#     Tokenizes and removes stopwords from the input text.
#     """
#     words = word_tokenize(text.lower())
#     stop_words = set(stopwords.words("english"))
#     return [word for word in words if word.isalnum() and word not in stop_words]

# def find_disease(user_input):
#     """
#     Finds the most relevant disease based on the user input symptoms.
#     This is a naive implementation that checks if any token from the input
#     is found in the disease's symptoms list.
#     """
#     user_symptoms = preprocess_text(user_input)
#     for disease, symptoms in disease_symptom_dict.items():
#         if any(symptom in symptoms for symptom in user_symptoms):
#             return disease
#     return "Disease not found. Please consult a doctor."

# def recommend_doctor(disease):
#     """
#     Recommends a doctor based on the disease.
#     The keys are in lowercase to allow case-insensitive matching.
#     """
#     specialists = {
#         "flu": "General Physician",
#         "diabetes": "Endocrinologist",
#         "asthma": "Pulmonologist",
#         "hypertension": "Cardiologist",
#         "migraine": "Neurologist"
#     }
#     return specialists.get(disease.lower(), "General Physician")

# def get_precaution(disease):
#     """
#     Returns the precautions/remedies for the given disease.
#     """
#     return disease_precaution_dict.get(disease, ["No specific precaution available."])

# # --------------------------
# # Testing (Optional)
# # --------------------------
# if __name__ == "__main__":
#     # Example usage
#     input_text = "I have fever and headache"
#     disease = find_disease(input_text)
#     doctor = recommend_doctor(disease)
#     precautions = get_precaution(disease)
    
#     print("Input:", input_text)
#     print("Predicted Disease:", disease)
#     print("Recommended Doctor:", doctor)
#     print("Precautions/Remedies:", precautions)

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

# Download required resources
nltk.download("punkt")
nltk.download("stopwords")

# Load and Process the CSV files as before
df_symptoms = pd.read_csv("DiseaseAndSymptoms.csv")
symptom_columns = [col for col in df_symptoms.columns if col.startswith("Symptom_")]
df_symptoms["symptoms"] = df_symptoms[symptom_columns].apply(
    lambda row: [str(s).strip().lower() for s in row if pd.notna(s)], axis=1
)
disease_symptom_dict = dict(zip(df_symptoms["Disease"], df_symptoms["symptoms"]))

df_precaution = pd.read_csv("Disease_precaution.csv")
precaution_columns = [col for col in df_precaution.columns if col.startswith("Precaution_")]
df_precaution["precautions"] = df_precaution[precaution_columns].apply(
    lambda row: [str(p).strip() for p in row if pd.notna(p)], axis=1
)
disease_precaution_dict = dict(zip(df_precaution["Disease"], df_precaution["precautions"]))

def preprocess_text(text):
    """Tokenizes and removes stopwords from the input text."""
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word.isalnum() and word not in stop_words]

def find_disease(user_input):
    """
    Attempts to find the most relevant disease based on user input symptoms.
    Improved logic checks for:
      - Direct substring matching with the full input
      - N-gram matching (e.g., bi-grams) for multi-word symptoms.
    """
    lower_input = user_input.lower().strip()
    tokens = preprocess_text(user_input)
    bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]

    for disease, symptoms in disease_symptom_dict.items():
        for symptom in symptoms:
            # Check if the symptom (multi-word or single word) appears as a substring in the input
            if symptom in lower_input:
                return disease
            # Also check if any token equals the symptom (in case the CSV has single-word symptoms)
            if symptom in tokens:
                return disease
            # Check for bigram matches
            if symptom in bigrams:
                return disease

    return "Disease not found. Please consult a doctor."

def recommend_doctor(disease):
    """
    Recommends a doctor based on the disease.
    """
    specialists = {
        "flu": "General Physician",
        "diabetes": "Endocrinologist",
        "asthma": "Pulmonologist",
        "hypertension": "Cardiologist",
        "migraine": "Neurologist"
    }
    return specialists.get(disease.lower(), "General Physician")

def get_precaution(disease):
    """
    Returns the precautionary recommendations for the given disease.
    """
    return disease_precaution_dict.get(disease, ["No specific precaution available."])

# Optional testing block
if __name__ == "__main__":
    input_text = "I have chest pain and shortness of breath"
    disease = find_disease(input_text)
    doctor = recommend_doctor(disease)
    precautions = get_precaution(disease)
    
    print("Input:", input_text)
    print("Predicted Disease:", disease)
    print("Recommended Doctor:", doctor)
    print("Precautions/Remedies:", precautions)
