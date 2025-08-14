import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)

    data = {
        'Age': np.random.randint(10, 80, num_samples),
        'Gender': np.random.choice(['Male', 'Female'], num_samples),
        'Cough': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Fever': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
        'Weight_Loss': np.random.choice([0, 1], num_samples, p=[0.75, 0.25]),
        'Night_Sweats': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'Fatigue': np.random.choice([0, 1], num_samples, p=[0.65, 0.35]),
        'Chest_Pain': np.random.choice([0, 1], num_samples, p=[0.85, 0.15]),
        'Shortness_of_Breath': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'Contact_with_TB': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'HIV_Status': np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
        'Diabetes': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'Smoking': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'Alcohol_Consumption': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'TB_Diagnosis': np.zeros(num_samples, dtype=int)  # Default to 0 (No TB)
    }

    df = pd.DataFrame(data)

    # Introduce some correlation for TB_Diagnosis
    # For simplicity, let's say if a patient has a combination of cough, fever, weight loss, and contact with TB, they are more likely to have TB
    df.loc[
        (df['Cough'] == 1) &
        (df['Fever'] == 1) &
        (df['Weight_Loss'] == 1) &
        (df['Contact_with_TB'] == 1),
        'TB_Diagnosis'
    ] = np.random.choice([0, 1], df[(df['Cough'] == 1) & (df['Fever'] == 1) & (df['Weight_Loss'] == 1) & (df['Contact_with_TB'] == 1)].shape[0], p=[0.2, 0.8])

    # Add some noise to other symptoms for TB_Diagnosis
    df.loc[
        (df['Night_Sweats'] == 1) | 
        (df['Fatigue'] == 1) | 
        (df['Shortness_of_Breath'] == 1),
        'TB_Diagnosis'
    ] = np.random.choice([0, 1], df[(df['Night_Sweats'] == 1) | (df['Fatigue'] == 1) | (df['Shortness_of_Breath'] == 1)].shape[0], p=[0.6, 0.4])

    # Ensure some positive cases
    num_tb_cases = int(num_samples * 0.15) # Approximately 15% of cases will be TB
    tb_indices = np.random.choice(df[df['TB_Diagnosis'] == 0].index, num_tb_cases, replace=False)
    df.loc[tb_indices, 'TB_Diagnosis'] = 1

    return df

if __name__ == '__main__':
    synthetic_data = generate_synthetic_data()
    synthetic_data.to_csv('tuberculosis_data.csv', index=False)
    print('Données synthétiques générées et sauvegardées dans tuberculosis_data.csv')