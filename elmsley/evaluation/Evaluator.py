import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
import os
import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self):
        pass

    def evaluate_model(self, model, path, x_train, y_train, x_test, y_test, x_external, y_external, label_encoder):
        model_folder = os.path.join(path, model.__class__.__name__)
        external_folder = os.path.join(path, 'external_db', model.__class__.__name__)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(external_folder, exist_ok=True)

        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
        y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)
        y_train_decoded = label_encoder.inverse_transform(y_train)
        y_test_decoded = label_encoder.inverse_transform(y_test)

        print(f"{model.__class__.__name__} - Classification Report (Train):")
        print(classification_report(y_train_decoded, y_pred_train_decoded))
        print(f"{model.__class__.__name__} - Classification Report (Test):")
        print(classification_report(y_test_decoded, y_pred_test_decoded))
        print(f"Recall: {recall_score(y_test_decoded, y_pred_test_decoded, average='macro')}")
        print(f"Precision: {precision_score(y_test_decoded, y_pred_test_decoded, average='macro')}")
        print(f"F1 score: {f1_score(y_test_decoded, y_pred_test_decoded, average='macro')}")

        train_report = classification_report(y_train_decoded, y_pred_train_decoded, output_dict=True)
        test_report = classification_report(y_test_decoded, y_pred_test_decoded, output_dict=True)

        train_report_df = pd.DataFrame(train_report).transpose()
        test_report_df = pd.DataFrame(test_report).transpose()

        train_report_df.to_csv(f"{model_folder}/{model.__class__.__name__}_train_classification_report.csv")
        test_report_df.to_csv(f"{model_folder}/{model.__class__.__name__}_test_classification_report.csv")

        print('Confusion matrix calculation')
        print(confusion_matrix(y_test, y_pred_test))
        cm_train = confusion_matrix(y_train_decoded, y_pred_train_decoded)
        cm_test = confusion_matrix(y_test_decoded, y_pred_test_decoded)

        cm_train_df = pd.DataFrame(cm_train, index=label_encoder.classes_, columns=label_encoder.classes_)
        cm_test_df = pd.DataFrame(cm_test, index=label_encoder.classes_, columns=label_encoder.classes_)

        cm_train_df.to_csv(os.path.join(model_folder, f"{model.__class__.__name__}_train_confusion_matrix.csv"))
        cm_test_df.to_csv(os.path.join(model_folder, f"{model.__class__.__name__}_test_confusion_matrix.csv"))

        if x_external and y_external:

            y_external_decoded = label_encoder.inverse_transform(y_external)
            y_pred_external = model.predict(x_external)
            y_pred_external_decoded = label_encoder.inverse_transform(y_pred_external)

            print(f"{model.__class__.__name__} - Classification Report (External):")
            print(classification_report(y_external_decoded, y_pred_external_decoded))
            print(f"Recall: {recall_score(y_external_decoded, y_pred_external_decoded, average='macro')}")
            print(f"Precision: {precision_score(y_external_decoded, y_pred_external_decoded, average='macro')}")
            print(f"F1 score: {f1_score(y_external_decoded, y_pred_external_decoded, average='macro')}")

            external_report_df = pd.DataFrame(classification_report(y_external_decoded, y_pred_external_decoded, output_dict=True)).transpose()
            external_report_df.to_csv(os.path.join(external_folder, f"{model.__class__.__name__}_external_classification_report.csv"))

            print('Confusion matrix calculation (External):')
            print(confusion_matrix(y_external, y_pred_external))

            cm_external = confusion_matrix(y_external, y_pred_external)
            cm_external_df = pd.DataFrame(cm_external, index=np.unique(y_external), columns=np.unique(y_external))
            cm_external_df.to_csv(os.path.join(external_folder, f"{model.__class__.__name__}_external_confusion_matrix.csv"))
