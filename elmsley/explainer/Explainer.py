import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from elmsley.internal.utils import elbow_method

class Explainer:
    def __init__(self):
        pass

    def explain(self, path, model, x_train, y_test_decoded, x_test, y_pred_test, x_external, y_external_decoded, y_pred_external, label_encoder):

        print('Start explanation')
        k_optimal = elbow_method(x_train)
        try:
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(x_train, k_optimal))
        except:
            explainer = shap.TreeExplainer(model)

        kmeans = KMeans(n_clusters=k_optimal, random_state=42)
        kmeans.fit(x_train)


        x_test_with_clusters = x_test.copy()
        x_test_with_clusters['cluster'] = kmeans.predict(x_test)
        min_subset_test = min(x_test_with_clusters['cluster'].value_counts())
        subset_test = x_test_with_clusters.groupby('cluster').apply(lambda x: x.sample(n=min_subset_test, random_state=42)).reset_index(drop=True)

        shap_values_ex = explainer(subset_test.drop(columns=['cluster']))
        subset_indices = subset_test.index
        y_subset_test = y_test_decoded[subset_indices]
        y_subset_pred = y_pred_test[subset_indices]

        shap_means = []
        for i in range(shap_values_ex.values.shape[2]):
            shap_means.append(np.mean(np.abs(shap_values_ex.values[:, :, i]), axis=0))

        class_names = label_encoder.inverse_transform(model.classes_)
        df = pd.DataFrame({name: mean for name, mean in zip(class_names, shap_means)})


        x_external_with_clusters = x_external.copy()
        x_external_with_clusters['cluster'] = kmeans.predict(x_external)
        min_subset_external = min(x_external_with_clusters['cluster'].value_counts())
        subset_external = x_external_with_clusters.groupby('cluster').apply(lambda x: x.sample(n=min_subset_external, random_state=42)).reset_index(drop=True)


        subset_external_indices = subset_external.index
        y_subset_external = y_external_decoded[subset_external_indices]
        y_subset_pred_external = y_pred_external[subset_external_indices]

        shap_values_external = explainer(subset_external.drop(columns=['cluster']))

        shap_means_external = []
        for i in range(shap_values_external.values.shape[2]):
            shap_means_external.append(np.mean(np.abs(shap_values_ex.values[:, :, i]), axis=0))

        df_external = pd.DataFrame({name: mean for name, mean in zip(class_names, shap_means_external)})


        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        df.plot.bar(ax=ax)
        ax.set_ylabel("Means SHAP", size=30)
        ax.set_xticklabels(x_train.columns, rotation=45, size=20)
        ax.legend(fontsize=30)
        plt.savefig(f"{model_folder}/SHAP_Bar_{model.__class__.__name__}.pdf",
                    bbox_inches="tight")
        plt.close()

        self._save_waterfalls(shap_values_ex, y_subset_test, y_subset_pred, model, label_encoder, model_folder)

        self._save_waterfalls(shap_values_external, y_subset_external, y_subset_pred_external, model, label_encoder, external_folder)



    def _save_waterfalls(self, shap_values, y_true, y_pred, model, label_encoder, folder):
        correctly_classified = (y_true == y_pred)
        incorrectly_classified = ~correctly_classified

        if np.any(correctly_classified):
            idx_correct = np.where(correctly_classified)[0][0]
            for label in model.classes_:
                shap.plots.waterfall(shap_values[idx_correct, :, label], show=False)
                plt.title(f"SHAP Values - Correctly Classified")
                filename = f"{folder}/SHAP_Waterfall_Correct_{label_encoder.inverse_transform([label])[0]}_{model.__class__.__name__}.pdf"
                plt.savefig(filename, bbox_inches="tight")
                plt.close()

        if np.any(incorrectly_classified):
            idx_incorrect = np.where(incorrectly_classified)[0][0]
            for label in model.classes_:
                shap.plots.waterfall(shap_values[idx_incorrect, :, label], show=False)
                plt.title(f"SHAP Values - Incorrectly Classified")
                filename = f"{folder}/SHAP_Waterfall_Incorrect_{label_encoder.inverse_transform([label])[0]}_{model.__class__.__name__}.pdf"
                plt.savefig(filename, bbox_inches="tight")
                plt.close()