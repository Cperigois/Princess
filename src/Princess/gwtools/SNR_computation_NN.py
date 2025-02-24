import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load
import os
from matplotlib.backends.backend_pdf import PdfPages


def process_NN_models():
    # Load file
    path = '../Run/GC_analysis/Astro_Models/Catalogs/'
    input_file = path + "GC_ngng_oleary_noclusterevolv.dat"
    output_columns = [
        "LO4", "HO4", "VO4", "KO4",
        "LO5", "HO5", "VO5", "KO5",
        "ET10km", "CE_H_20km", "CE_H_40km", "CE_L_40km"
    ]

    # Loading data
    data = pd.read_csv(input_file, sep="\t", index_col = None)

    # Filtrer les lignes où CE_L_40km est égal à 0.0
    data = data[data["CE_L_40km"] != 0.0]
    print(data.describe())

    # Entry columns
    input_columns = ["Mc", "q", "Dl"]

    # preparation of outputs directories
    base_dir = "../AuxiliarxFiles"
    snr_nn_dir = os.path.join(base_dir, "SNR_NN_computation_Reg")
    models_dir = os.path.join(snr_nn_dir, "models")
    results_dir = os.path.join(snr_nn_dir, "results")

    # Création des répertoires si nécessaires
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Folders already exists :\n- {models_dir}\n- {results_dir}")

    # Run for each detector
    for output_column in output_columns:
        print(f"Training model for {output_column}...")

        # Prepare data
        X = data[input_columns].values
        y = data[output_column].values

        # Separate the training and valid datasets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # REGRESSION MODEL #
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # Sauvegarde du modèle
        model_filename = f"{models_dir}/{output_column}_optimal.joblib"
        dump(model, model_filename)
        print(f"Modèle sauvegardé sous : {model_filename}")

        # SEQUENTIAL MODEL #
        # model = Sequential([
        #     Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        #     Dense(32, activation="relu"),
        #     Dense(16, activation="relu"),
        #     Dense(1)  # Une seule sortie pour prédire la colonne cible
        # ])
        #
        # model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
        #
        # # Model training
        # history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)

        # Model validation
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        #relative error
        relative_errors = np.abs((y_val - y_pred) / y_val)  # Éviter la division par zéro si y_val contient 0
        rme = np.mean(relative_errors[np.isfinite(relative_errors)])  # Ignorer les infinis et NaN


        # Save statistics
        stats_file = f"{results_dir}/{output_column}_valid.txt"
        with open(stats_file, "w") as f:
            f.write(f"Model: {output_column}\n")
            f.write(f"Mean Squared Error (MSE): {mse}\n")
            f.write(f"Mean Absolute Error (MAE): {mae}\n")
            f.write(f"R-squared (R2): {r2}\n")
            f.write(f"Relative Mean Error (RME): {rme}\n")
            #f.write(f"History:\n")
            #for key in history.history.keys():
            #    f.write(f"{key}: {history.history[key]}\n")

        predictions_df = pd.DataFrame({
            "Mc": X_val[:, 0],
            "q": X_val[:, 1],
            "dl": X_val[:, 2],
            "True_Values": y_val,
            "Predicted_Values": y_pred.flatten()
        })
        predictions_df.to_csv(os.path.join(results_dir, f"{output_column}_optimal_predictions.csv"), index=False)

        print(f"Model for {output_column} trained and saved.")

def Check_models() :
    output_columns = [
        "LO4", "HO4", "VO4", "KO4",
        "LO5", "HO5", "VO5", "KO5",
        "ET10km", "CE_H_20km", "CE_H_40km", "CE_L_40km"
    ]
    base_dir = "../AuxiliarxFiles"
    snr_nn_dir = os.path.join(base_dir, "SNR_NN_computation_Reg")
    models_dir = os.path.join(snr_nn_dir, "models")
    results_dir = os.path.join(snr_nn_dir, "results")
    with PdfPages(f'{results_dir}/Plots_Valid.pdf') as pdf:
        for name in output_columns :
            df = pd.read_csv(f"{results_dir}/{name}_optimal_predictions.csv", index_col = False)
            mae = np.abs(df['True_Values']-df['Predicted_Values'])
            mre = mae/df['True_Values']
            plt.scatter(df['True_Values'], mae )
            plt.title(label= name, fontsize=16)
            plt.xlabel('SNR True',fontsize=16)
            plt.ylabel('MAE',fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            pdf.savefig()
            plt.close()


            plt.scatter(df['True_Values'],mre )
            plt.title(label = name,fontsize=16)
            plt.xlabel('SNR True', fontsize = 16)
            plt.ylabel('MRE', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            plt.scatter(df['dl'], mre)
            plt.title(label = name,fontsize=16)
            plt.xlabel('Dl', fontsize = 16)
            plt.ylabel('MRE', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            pdf.savefig()
            plt.close()



if __name__ == "__main__":
    try:
        #process_NN_models()
        Check_models()
        print("La fonction bla a été exécutée avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'exécution de process_NN_models : {e}")

