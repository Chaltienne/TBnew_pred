from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

# Initialiser l'application Flask
app = Flask(__name__)

# Charger le modèle et le préprocesseur
model = joblib.load("tb_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Route principale avec formulaire
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            # Récupérer les données du formulaire
            input_data = {
                "Age": int(request.form["Age"]),
                "Gender": request.form["Gender"],
                "Cough": int(request.form["Cough"]),
                "Fever": int(request.form["Fever"]),
                "Weight_Loss": int(request.form["Weight_Loss"]),
                "Night_Sweats": int(request.form["Night_Sweats"]),
                "Fatigue": int(request.form["Fatigue"]),
                "Chest_Pain": int(request.form["Chest_Pain"]),
                "Shortness_of_Breath": int(request.form["Shortness_of_Breath"]),
                "Contact_with_TB": int(request.form["Contact_with_TB"]),
                "HIV_Status": int(request.form["HIV_Status"]),
                "Diabetes": int(request.form["Diabetes"]),
                "Smoking": int(request.form["Smoking"]),
                "Alcohol_Consumption": int(request.form["Alcohol_Consumption"])
            }

            df = pd.DataFrame([input_data])

            # Prétraitement
            X_processed = preprocessor.transform(df)
            if not isinstance(X_processed, np.ndarray):
                X_processed = X_processed.toarray()

            # Prédiction
            prediction = int(model.predict(X_processed)[0])
            probability = float(model.predict_proba(X_processed)[:, 1][0])

        except Exception as e:
            return f"Erreur lors de la prédiction: {e}"

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
