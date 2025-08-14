import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

if __name__ == "__main__":
    # Charger les données prétraitées
    try:
        X = pd.read_csv("tuberculosis_data_processed.csv")
        y = pd.read_csv("tuberculosis_labels.csv").squeeze()  # .squeeze() transforme en Series
        print("Données chargées avec succès.")
    except FileNotFoundError:
        print("Erreur : fichiers prétraités introuvables. Exécute d'abord data_preprocessing.py")
        exit()

    # Séparer en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Créer et entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions sur le jeu de test
    y_pred = model.predict(X_test)

    # Évaluer le modèle
    print("\n--- Évaluation du modèle ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nMatrice de confusion:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Sauvegarder le modèle entraîné
    joblib.dump(model, "tb_model.pkl")
    print("\nModèle sauvegardé sous 'tb_model.pkl'.")
