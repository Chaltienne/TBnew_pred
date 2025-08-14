import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def preprocess_data(df):
    # Vérifier la présence de la colonne cible
    if "TB_Diagnosis" not in df.columns:
        raise ValueError("La colonne 'TB_Diagnosis' est absente du dataset.")

    # Séparer caractéristiques (X) et cible (y)
    X = df.drop("TB_Diagnosis", axis=1)
    y = df["TB_Diagnosis"]

    # Identifier colonnes numériques et catégorielles
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    # Créer les transformateurs
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # <-- corrigé

    # Créer le préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Appliquer le préprocesseur
    X_processed = preprocessor.fit_transform(X)

    # Obtenir les noms des colonnes après encodage one-hot
    onehot_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    all_features = numerical_cols + list(onehot_features)

    # Transformer en DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=all_features)

    return X_processed_df, y, preprocessor

def perform_eda(df):
    print("\n--- Analyse Exploratoire des Données ---")
    print("Dimensions du dataset:", df.shape)
    print("\nInformations sur le dataset:\n")
    df.info()
    print("\nStatistiques descriptives:\n", df.describe())
    print("\nValeurs manquantes:\n", df.isnull().sum())

    # Distribution de la variable cible
    if "TB_Diagnosis" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x="TB_Diagnosis", data=df)
        plt.title("Distribution de la variable cible (TB_Diagnosis)")
        plt.show()

    # Matrice de corrélation pour colonnes numériques
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice de corrélation des caractéristiques numériques")
        plt.show()

if __name__ == "__main__":
    # Charger les données
    try:
        df = pd.read_csv("tuberculosis_data.csv")
        print("Données chargées avec succès.")
    except FileNotFoundError:
        print("Erreur : Le fichier 'tuberculosis_data.csv' est introuvable.")
        exit()

    perform_eda(df)
    X_processed, y, preprocessor = preprocess_data(df)
    print("\nDonnées prétraitées avec succès. Dimensions de X_processed:", X_processed.shape)
    print("\nExemple de données prétraitées:\n", X_processed.head())

    # Sauvegarder le préprocesseur
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("Préprocesseur sauvegardé sous 'preprocessor.pkl'.")

    # Sauvegarder les données prétraitées
    X_processed.to_csv("tuberculosis_data_processed.csv", index=False)
    y.to_csv("tuberculosis_labels.csv", index=False)
    print("Données prétraitées et labels sauvegardés.")
