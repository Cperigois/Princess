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


    def create_AI_model(self, data_lenght):

        import optuna

        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import train_test_split

        path = 'AuxiliaryFiles/SNR_pre_computation/' + self.reference

        # Make sure directories are created
        if not os.path.exists(path):
            os.mkdir(path)

        # Create a huge uniform fake population
        mass_params = {"low": 2, "high": 200}  # Masse in Msun
        z_params = {"low": 0.01, "high": 25}  # Redshift

        # Generate training dataset of 1 000 000 merging binaries
        population = generate_population(
            num_sources=data_lenght,
            mass_method="m1_m2",
            mass_distribution="uniform",
            z_distribution="uniform",
            mass_params=mass_params,
            z_params=z_params
        )
        save_population(population, "Run/temp/Training_dataset.dat")

        print(f"Start the SNR computation for {self.reference}")
        SNR_pycbc_catalog('Training_dataset', det_list=params['detector_list'].keys(),
                          waveform=params['detector_params']['types'][self.type]['waveform'], catalogue_path="Run/temp")

        X1_params = ['m1', 'm2', 'Dl', 'Mc_5-6', 'Dl-1', 'fmerg', 'Mcz_5-6', 'Dlz-1']

        # Training dataset
        Train_ds = pd.read_csv("Run/temp/Training_dataset.dat", sep='\t', index_col=None)
        Train_ds['Mc_5-6'] = Train_ds['Mc'] ** (5 / 6)
        Train_ds['Mcz_5-6'] = (Train_ds['Mc'] * (1 + Train_ds['z'])) ** (5 / 6)
        Train_ds['Dl-1'] = 1 / Train_ds['Dl']
        Train_ds['Dlz-1'] = (1 + Train_ds['z']) / (Train_ds['Dl'])

        print('Training dataset loaded!')

        X_train, X_test, y_train, y_test = train_test_split(Train_ds[X1_params], Train_ds[f'{self.reference}_pycbc'], test_size=0.2, random_state=42)

        # List to store results of all trials
        results_list = []

        # Run Optuna optimization
        study = optuna.create_study(direction="minimize")  # We want to minimize MAE
        study.optimize(objective, n_trials=100)  # Run 50 trials

        # Convert results to DataFrame
        results_df = pd.DataFrame(results_list)

        # Sort results by MAE (Best first)
        results_df = results_df.sort_values(by="mae", ascending=True)

        # Save results to a CSV file
        results_df.to_csv(f"{path}/Optuna_output.csv", index=False)
        print(f"\nOptimization results saved to '{path}/Optuna_output.csv'.")

        # Display top 5 best parameter sets
        print("\nTop 5 Best Trials:")
        print(results_df.head())

        # Train the final model with the best found parameters
        best_params = study.best_params
        best_model = HistGradientBoostingRegressor(**best_params)
        best_model.fit(X_train, y_train)

        # Generate validation dataset of 3 000 merging binaries
        population = generate_population(
            num_sources=3000,
            mass_method="m1_m2",
            mass_distribution="uniform",
            z_distribution="uniform",
            mass_params=mass_params,
            z_params=z_params
        )
        save_population(population, f"{path}/Validation_dataset.dat")

        print(f"Start the SNR computation for {self.reference} with pycbc")
        SNR_pycbc_catalog(f'Validation_dataset.dat', det_list=[self.name],
                          waveform=params['detector_params']['types'][self.type]['waveform'], catalogue_path=path)

        # Training dataset
        Test_ds = pd.read_csv(f"{path}/Test_dataset.dat", sep='\t', index_col=None)
        Test_ds['Mc_5-6'] = Test_ds['Mc'] ** (5 / 6)
        Test_ds['Mcz_5-6'] = (Test_ds['Mc'] * (1 + Test_ds['z'])) ** (5 / 6)
        Test_ds['Dl-1'] = 1 / Test_ds['Dl']
        Test_ds['Dlz-1'] = (1 + Test_ds['z']) / (Test_ds['Dl'])

        # Evaluate final model
        y_final_pred = best_model.predict(Test_ds[X1_params])
        final_mae = mean_absolute_error(Test_ds[f'{self.reference}_pycbc'], y_final_pred)

        Test_ds['perdicted_SNR'] = y_final_pred

        Test_ds.to_csv(f'{path}/Validation_dataset.dat', index = None, sep = '\t')

        model_path_L1 = f'{path}/{self.reference}.joblib'
        # Save the trained model
        joblib.dump(best_model, model_path_L1)
        print(f"Model saved as {model_path_L1}")

        self.training_report()
        #Remove training dataset
        os.remove('Run/temp/Training_dataset.csv')




    def training_report(self):

        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_absolute_error

        path = 'AuxiliaryFiles/SNR_pre_computation/' + self.reference
        results = pd.read_csv(f'{path}/Validation_dataset.dat', sep='\t', index_col=None)

        snr_thresholds = [8, 20, 50, 100]

        with PdfPages(f'{path}/Evaluation.pdf') as pdf:
            for snr in snr_thresholds:
                results[f"detect_{snr}"] = results[f"SNR_pycbc"] > snr
                results[f"predict_{snr}"] = results[f"predicted_SNR"] > snr
                results[f"error_{snr}"] = results[f"detect_{snr}"] != results[
                    f"predict_{snr}"]

                # Create scatter plot
                plt.figure(figsize=(10, 8))
                plt.xlabel(r"$\mathcal{M}_c$ (M$_\odot$)")
                plt.ylabel("Distance Dl (Mpc)")
                plt.title(f"Prediction Performance for SNR > {snr} ({self.reference})")

                # Correct predictions (green)
                plt.scatter(
                    results.loc[~results[f"error_{snr}"], "Mc"],
                    results.loc[~results[f"error_{snr}"], "Dl"],
                    color="green",
                    label="Correct Prediction",
                    alpha=0.5
                )

                # Misclassified predictions (red)
                plt.scatter(
                    results.loc[results[f"error_{snr}"], "Mc"],
                    results.loc[results[f"error_{snr}"], "Dl"],
                    color="red",
                    label="Misclassified",
                    alpha=0.5
                )

                plt.legend()
                plt.grid(True)

                # Save the figure to the PDF
                pdf.savefig()
                plt.close()
                # Additional plot: Mc vs. Distance with relative error coloring
            plt.figure(figsize=(8, 6))

            # Compute relative error on SNR prediction
            results[f"rel_error"] = np.abs(
                (results[f"predicted_SNR"] - results["SNR_pycbc"])
                / results["SNR_pycbc"]
            )

            # Scatter plot with color map based on relative error
            sc = plt.scatter(
                results["Mc"], results["Dl_Gpc"],
                c=results[f"rel_error"], cmap="plasma", alpha=0.7
            )

            # Add color bar
            cbar = plt.colorbar(sc)
            cbar.set_label("Relative SNR Prediction Error")

            # Labels and title
            plt.xlabel(r"$\mathcal{M}_c$ in M$_\odot$")
            plt.ylabel("Distance Dl in Gpc")
            plt.title("Relative SNR Error")
            plt.grid(True)

            # Save plot to PDF
            pdf.savefig()
            plt.close()

        print(f"Performance plots saved as {path}/evaluation.pdf")

        # Write performance report
        with open(f'{path}/performance_report.txt', "w") as f:
            f.write(f"Model Evaluation: {self.reference}\n")
            error = mean_absolute_error(results[f"predicted_SNR"], results[f"SNR_pycbc"])
            f.write(f"Mean Absolute Error (MAE): {error:.2f}\n\n")
            for snr in snr_thresholds:
                error_rate = results[f"error_{snr}"].mean() * 100
                size = len(results[results[f"detect_{snr}"] > snr])
                f.write(f"Error rate for SNR > {snr}: {error_rate:.2f}% among {size} events truly detected\n")

        print(f"Performance report saved as {path}/performance_report.txt")


if __name__ == "__main__":
    try:
        #process_NN_models()
        Check_models()
        print("La fonction bla a été exécutée avec succès.")
    except Exception as e:
        print(f"Erreur lors de l'exécution de process_NN_models : {e}")

