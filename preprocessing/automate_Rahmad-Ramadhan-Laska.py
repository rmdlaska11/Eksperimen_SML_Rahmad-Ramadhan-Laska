import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

def preprocessing_pipeline(filepath):
    # Buat folder model jika belum ada
    os.makedirs('preprocessing/model', exist_ok=True)

    # Load data
    df = pd.read_csv(filepath)

    # Tangani outlier
    def count_and_handle_outliers(df, method='cap'):    
        df_out = df.copy()
        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            if method == 'remove':
                df_out = df_out[~outlier_mask]
            elif method == 'cap':
                df_out[col] = df_out[col].clip(lower, upper)
        return df_out

    df = count_and_handle_outliers(df, method='cap')

    # Pisahkan fitur dan target
    X = df.drop(columns='Weather Type')
    y = df['Weather Type']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Scaling numerik
    def scale_data(X_train, X_test):
        scaler = MinMaxScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

        joblib.dump(scaler, 'preprocessing/model/minmax_scaler.joblib')
        return X_train_scaled, X_test_scaled

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Encode fitur kategorikal
    def encode_features(X_train, X_test):
        X_train_enc = X_train.copy()
        X_test_enc = X_test.copy()
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            encoder = LabelEncoder()
            X_train_enc[col] = encoder.fit_transform(X_train[col])
            X_test_enc[col] = encoder.transform(X_test[col])
            joblib.dump(encoder, f'preprocessing/model/encoder_{col}.joblib')

        return X_train_enc, X_test_enc

    X_train_encoded, X_test_encoded = encode_features(X_train_scaled, X_test_scaled)

    # Encode label
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    joblib.dump(label_encoder, 'preprocessing/model/encoder_target.joblib')

    # Gabungkan hasil
    train_final = X_train_encoded.copy()
    train_final['Weather Type'] = y_train_enc

    test_final = X_test_encoded.copy()
    test_final['Weather Type'] = y_test_enc

    return train_final, test_final

# === Bagian utama untuk automatisasi ===
def main():
    # Path file input dan output
    input_path = "weathertype_raw/weather_classification_data.csv"
    output_dir = "preprocessing"
    os.makedirs(output_dir, exist_ok=True)

    output_train = os.path.join(output_dir, "train_data.csv")
    output_test = os.path.join(output_dir, "test_data.csv")

    # Jalankan pipeline dan simpan hasil
    train_final, test_final = preprocessing_pipeline(input_path)
    train_final.to_csv(output_train, index=False)
    test_final.to_csv(output_test, index=False)
    print(f"✅ Preprocessing selesai. Dataset disimpan di:\n- {output_train}\n- {output_test}")

if __name__ == "__main__":
    main()
