import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# TOUT NOTRE CODE EST FAIT DANS CE FILE
# DECOMMENTER LA PARTIE A EXECUTER ET COMMENTER LE RESTE

features_file = '/home/koutsodi/5A/apprentisaage-supervise/alt_acsincome_ca_features_85(1).csv'
labels_file = '/home/koutsodi/5A/apprentisaage-supervise/alt_acsincome_ca_labels_85.csv'

features = pd.read_csv(features_file)
labels = pd.read_csv(labels_file)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
joblib.dump(scaler, 'scaler.joblib')


#------------------------------------------------------------------------------------------------------------
# Validation Croisee

#RandomForestClassifier
# rf_model = RandomForestClassifier() #parametres par defaut

# rf_predictions = cross_val_predict(rf_model, X_train_scaled, y_train.values.ravel(), cv=5) 

# rf_accuracy = cross_val_score(rf_model, X_train_scaled, y_train.values.ravel(), cv=5, scoring='accuracy')
# print(f'Accuracy moyen Random Forest : {rf_accuracy.mean():.4f}')

# print("Classification Report (Random Forest) :\n", classification_report(y_train, rf_predictions))
# print("Confusion Matrix (Random Forest) :\n", confusion_matrix(y_train, rf_predictions))

#------------------------------------------------------------------------------------------------------------

# #AdaBoostClassifier
# ada_model = AdaBoostClassifier()
# ada_predictions = cross_val_predict(ada_model, X_train_scaled, y_train.values.ravel(), cv=5)

# ada_accuracy = cross_val_score(ada_model, X_train_scaled, y_train.values.ravel(), cv=5, scoring='accuracy')
# print(f'Accuracy moyen AdaBoost : {ada_accuracy.mean():.4f}')

# print("Classification Report (AdaBoost) :\n", classification_report(y_train, ada_predictions))
# print("Confusion Matrix (AdaBoost) :\n", confusion_matrix(y_train, ada_predictions))

#------------------------------------------------------------------------------------------------------------

# #Gradient Boosting
# gb_model = GradientBoostingClassifier()
# gb_predictions = cross_val_predict(gb_model, X_train_scaled, y_train.values.ravel(), cv=5)
# gb_accuracy = cross_val_score(gb_model, X_train_scaled, y_train.values.ravel(), cv=5, scoring='accuracy')
# print(f'Accuracy moyen Gradient Boosting : {gb_accuracy.mean():.4f}')

# print("Classification Report (Gradient Boosting) :\n", classification_report(y_train, gb_predictions))
# print("Confusion Matrix (Gradient Boosting) :\n", confusion_matrix(y_train, gb_predictions))

#------------------------------------------------------------------------------------------------------------

#Random forest with improvement
# rf_model = RandomForestClassifier(random_state=42)
# param_grid = {
#     'n_estimators': [50, 100],  
#     'max_depth': [None, 10],       
#     'min_samples_split': [2, 5],       
#     'min_samples_leaf': [1, 2]        
# }
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=10)
# grid_search.fit(X_train_scaled, y_train)

# print("Best parameters : ", grid_search.best_params_)
# y_pred = grid_search.best_estimator_.predict(X_test_scaled)
# print("Classification Report :\n", classification_report(y_test, y_pred))
# joblib.dump(grid_search.best_estimator_,'RandomForest_BestModel_082.joblib')

##------------------------------------------------------------------------------------------------------------

##AdaBoost with improvement
# adaboost_model = AdaBoostClassifier(random_state=42)
# param_grid_adaboost = {
#     'n_estimators': [50, 100], 
#     'learning_rate': [0.01, 0.1] 
# }
# grid_search_adaboost = GridSearchCV(estimator=adaboost_model, param_grid=param_grid_adaboost, cv=5, scoring='accuracy', verbose=10)
# grid_search_adaboost.fit(X_train_scaled, y_train)
# print("Best parameters : ", grid_search_adaboost.best_params_)
# y_pred_adaboost = grid_search_adaboost.best_estimator_.predict(X_test_scaled)
# print("Classification Report pour AdaBoost :\n", classification_report(y_test, y_pred_adaboost))
# joblib.dump(grid_search_adaboost.best_estimator_,'AdaBoost_BestModel_080.joblib')

##------------------------------------------------------------------------------------------------------------

##GradientBoost with improvement
# gb_model = GradientBoostingClassifier(random_state=42)
# param_grid_gb = {
#     'n_estimators': [50, 100],   
#     'learning_rate': [0.01, 0.1], 
#     'max_depth': [3, 5]         
# }
# grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, cv=5, scoring='accuracy', verbose=10)
# grid_search_gb.fit(X_train_scaled, y_train)
# print("Best parameters : ", grid_search_gb.best_params_)
# y_pred_gb = grid_search_gb.best_estimator_.predict(X_test_scaled)
# print("Classification Report pour GradientBoosting :\n", classification_report(y_test, y_pred_gb))
# joblib.dump(grid_search_gb.best_estimator_,'GradientBoost_BestModel_082.joblib')

#------------------------------------------------------------------------------------------------------------

# RandomForestClassifier with best parameters

rf_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/RandomForest_BestModel_082.joblib")
y_predict_rf = rf_model.predict(X_test_scaled) 

print("Performance sur California (Random Forest) :")
print(classification_report(y_test, y_predict_rf))
print("Matrice de confusion (Random Forest, California) :\n", confusion_matrix(y_test, y_predict_rf))

#------------------------------------------------------------------------------------------------------------

# # AdaBoostClassifier with best parameters

ab_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/AdaBoost_BestModel_080.joblib")
y_predict_ab = ab_model.predict(X_test_scaled) 

print("Performance sur California (AdaBoost) :")
print(classification_report(y_test, y_predict_ab))
print("Matrice de confusion (AdaBoost, California) :\n", confusion_matrix(y_test, y_predict_ab))

# ------------------------------------------------------------------------------------------------------------

# Gradient Boosting with best parameters

gb_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/GradientBoost_BestModel_082.joblib")
y_predict_gb = gb_model.predict(X_test_scaled) 

print("Performance sur California (GradientBoosting) :")
print(classification_report(y_test, y_predict_gb))
print("Matrice de confusion (GradientBoosting, California) :\n", confusion_matrix(y_test, y_predict_gb))

#------------------------------------------------------------------------------------------------------------

# Comparaison des méthodes 

report_rf = classification_report(y_test, y_predict_rf, output_dict=True)
report_ab = classification_report(y_test, y_predict_ab, output_dict=True)
report_gb = classification_report(y_test, y_predict_gb, output_dict=True)

metrics_df = pd.DataFrame({
    "Model": ["Random Forest", "AdaBoost", "Gradient Boosting"],
    "Accuracy": [report_rf["accuracy"], report_ab["accuracy"], report_gb["accuracy"]],
    "Precision": [report_rf["weighted avg"]["precision"], report_ab["weighted avg"]["precision"], report_gb["weighted avg"]["precision"]],
    "Recall": [report_rf["weighted avg"]["recall"], report_ab["weighted avg"]["recall"], report_gb["weighted avg"]["recall"]],
    "F1 Score": [report_rf["weighted avg"]["f1-score"], report_ab["weighted avg"]["f1-score"], report_gb["weighted avg"]["f1-score"]]
})

print(metrics_df)

#------------------------------------------------------------------------------------------------------------

# Méthode de stacking de notre choix avec parametres par defaut

# Validation Croisee

# base_models = [
#   ('rf', RandomForestClassifier(random_state=42)),
# ('gb', GradientBoostingClassifier(random_state=42))
# ]

# meta_model = LogisticRegression(random_state=42)
# stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)
# stacking_predictions = cross_val_predict(stacking_clf, X_train_scaled, y_train, cv=5)

# #Validation croisée : Accuracy
# stacking_accuracy = cross_val_score(stacking_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
# print(f"\n Accuracy moyen StackingClassifier : {stacking_accuracy.mean():.4f}")

# print("\n Classification Report (StackingClassifier) :")
# print(classification_report(y_train, stacking_predictions))

# print("\n Confusion Matrix (StackingClassifier) :")
# print(confusion_matrix(y_train, stacking_predictions))

#------------------------------------------------------------------------------------------------------------

# Stacking looking for best parameters
# param_grid_stacking = {
  # 'rf__n_estimators': [50, 100],
  # 'rf__max_depth': [None, 10],
  # 'gb__n_estimators': [50, 100],
  # 'gb__learning_rate': [0.01, 0.1],
  # 'final_estimator__C': [0.1, 1, 10]
# }

# grid_search = GridSearchCV(estimator=stacking_clf, param_grid=param_grid_stacking, cv=5, scoring='accuracy', verbose=10)

# grid_search.fit(X_train_scaled, y_train)

# print("\n Best parameters :")
# print(grid_search.best_params_)

# y_pred = grid_search.best_estimator_.predict(X_test_scaled)

# print("\n Classification Report (StackingClassifier) :")
# print(classification_report(y_test, y_pred))

# print("\nConfusion Matrix (StackingClassifier) :")
# print(confusion_matrix(y_test, y_pred))
# joblib.dump(grid_search.best_estimator_, 'Stacking_BestModel_082.joblib')

#------------------------------------------------------------------------------
# Partie 3 Question 5

# rf_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/RandomForest_BestModel_082.joblib")
# gb_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/GradientBoost_BestModel_082.joblib")
# ab_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/AdaBoost_BestModel_080.joblib")

# features_file_co = '/home/koutsodi/5A/apprentisaage-supervise/acsincome_co_features.csv'
# labels_file_co = '/home/koutsodi/5A/apprentisaage-supervise/acsincome_co_label.csv'

# features_file_ne = '/home/koutsodi/5A/apprentisaage-supervise/acsincome_ne_features.csv'
# labels_file_ne = '/home/koutsodi/5A/apprentisaage-supervise/acsincome_ne_labelTP2.csv'

# features_co = pd.read_csv(features_file_co)
# labels_co = pd.read_csv(labels_file_co)

# features_ne = pd.read_csv(features_file_ne)
# labels_ne = pd.read_csv(labels_file_ne)


# X_scaled_co = scaler.transform(features_co)
# X_scaled_ne = scaler.transform(features_ne)

# #RandomForest
# y_predict_rf_co = rf_model.predict(X_scaled_co) 
# y_predict_rf_ne = rf_model.predict(X_scaled_ne) 

# print("Performance sur Colorado (Random Forest) :")
# print(classification_report(labels_co, y_predict_rf_co))
# print("Matrice de confusion (Random Forest, Colorado) :\n", confusion_matrix(labels_co, y_predict_rf_co))

# print("Performance sur Nevada (Random Forest) :")
# print(classification_report(labels_ne, y_predict_rf_ne))
# print("Matrice de confusion (Random Forest, Nevada) :\n", confusion_matrix(labels_ne, y_predict_rf_ne))

# AdaBoost
# y_predict_ab_co = ab_model.predict(X_scaled_co) 
# y_predict_ab_ne = ab_model.predict(X_scaled_ne) 

# print("Performance sur Colorado (AdaBoost) :")
# print(classification_report(labels_co, y_predict_ab_co))
# print("Matrice de confusion (AdaBoost, Colorado) :\n", confusion_matrix(labels_co, y_predict_ab_co))

# print("Performance sur Nevada (AdaBoost) :")
# print(classification_report(labels_ne, y_predict_ab_ne))
# print("Matrice de confusion (AdaBoost, Nevada) :\n", confusion_matrix(labels_ne, y_predict_ab_ne))

#GradientBoosting

# y_predict_gb_co = ab_model.predict(X_scaled_co) 
# y_predict_gb_ne = ab_model.predict(X_scaled_ne) 

# print("Performance sur Colorado (GradientBoosting) :")
# print(classification_report(labels_co, y_predict_gb_co))
# print("Matrice de confusion (GradientBoosting, Colorado) :\n", confusion_matrix(labels_co, y_predict_gb_co))

# print("Performance sur Nevada (GradientBoosting) :")
# print(classification_report(labels_ne, y_predict_gb_ne))
# print("Matrice de confusion (GradientBoosting, Nevada) :\n", confusion_matrix(labels_ne, y_predict_gb_ne))

# #------------------------------------------------------------------------------------------------------------

# # 4.1 Question 1 -----------------------------------------------------------------------------------------------------
# # Corrélations initiales 
# features['PINCP']=labels # Concaténation de la colonne PINCP de labels dans features
# features['PINCP'] = LabelEncoder().fit_transform(features['PINCP']) # Pour avoir 0 et 1 au lieu de False et True 
# features.fillna(features.median(), inplace=True)# Pour remplacer NA avec la médiane
# numeric_features = features.drop(columns=['PINCP']).columns  
# correlations = {}
# for col in numeric_features:
#    corr, _ = pearsonr(features[col], features['PINCP'])
#    correlations[col] = corr
# print("Numerical feature correlations with PINCP:")
# for feature, corr in correlations.items():
#     print(f"{feature}: {corr:.2f}")
# plt.figure(figsize=(10, 8))
# corr_matrix = features.corr()  # Includes PINCP
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
# plt.title("Matrice de corrélations")
# plt.show()

# # Corrélations produites par les méthodes d'apprentissage

# rf_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/RandomForest_BestModel_082.joblib")
# gb_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/GradientBoost_BestModel_082.joblib")
# ab_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/AdaBoost_BestModel_080.joblib")

#Matrice de corrélation avec RandomForest
# y_predict_rf = rf_model.predict(X_test_scaled) 
# X_test_with_predictions = X_test.copy()  
# X_test_with_predictions['Prediction'] = y_predict_rf  
# corr_matrix = X_test_with_predictions.corr()  

# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
# plt.title("Matrice de corrélations entre les caractéristiques et les prédictions du modèle Random Forest")
# plt.show()

#Matrice de corrélation avec Gradient Boosting
# y_predict_gb = gb_model.predict(X_test_scaled) 
# X_test_with_predictions = X_test.copy()  
# X_test_with_predictions['Prediction'] = y_predict_gb  
# corr_matrix = X_test_with_predictions.corr()  

# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
# plt.title("Matrice de corrélations entre les caractéristiques et les prédictions du modèle Gradient Boosting")
# plt.show()

#Matrice de corrélation avec Ada Boost
# y_predict_ab = ab_model.predict(X_test_scaled) 
# X_test_with_predictions = X_test.copy()  
# X_test_with_predictions['Prediction'] = y_predict_ab  
# corr_matrix = X_test_with_predictions.corr()  

# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
# plt.title("Matrice de corrélations entre les caractéristiques et les prédictions du modèle AdaBoost")
# plt.show()

# print("QUESTION 2 -----------------------------------------------------------------------------")

# Random Forest

# importances_rf = rf_model.feature_importances_
# importance_rf_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': importances_rf
# })
# importance_rf_df = importance_rf_df.sort_values(by='Importance', ascending=False)
# plt.figure(figsize=(10, 6))
# plt.barh(importance_rf_df['Feature'], importance_rf_df['Importance'])
# plt.xlabel('Feature Importance')
# plt.title('Feature Importance from Random Forest Model')
# plt.show()

# # Ada Boost 

# importances_ab = ab_model.feature_importances_
# importance_ab_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': importances_ab
# })
# importance_ab_df = importance_ab_df.sort_values(by='Importance', ascending=False)
# plt.figure(figsize=(10, 6))
# plt.barh(importance_ab_df['Feature'], importance_ab_df['Importance'])
# plt.xlabel('Feature Importance')
# plt.title('Feature Importance from Ada Boost Model')
# plt.show()

# # Gradient Boost 

# importances_gb = gb_model.feature_importances_
# importance_gb_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': importances_gb
# })
# importance_gb_df = importance_gb_df.sort_values(by='Importance', ascending=False)
# plt.figure(figsize=(10, 6))
# plt.barh(importance_gb_df['Feature'], importance_gb_df['Importance'])
# plt.xlabel('Feature Importance')
# plt.title('Feature Importance from Gradient Boost Model')
# plt.show()


# #------------------------------------------------------------------------------------------------------------

# # 4.2 Question 1 -----------------------------------------------------------------------------------------------------
# print("QUESTION 1 -----------------------------------------------------------------------------")
# total = len(labels) # Nombre total de personnes
# sup_50000 = labels[labels['PINCP'] == True]
# taux = len(sup_50000)/total

# print("\n Taux de personnes ayant un salaire supérieur à 50 000 dollars : ")
# print(taux)

# features['PINCP']=labels # Concaténation de la colonne PINCP de labels dans features

# # Hommes

# homme_et_sup_50000 = features[(features['SEX'] == 1.0) & (features['PINCP'] == True)]
# taux_homme_sup_50000 = len(homme_et_sup_50000) / len(features[features['SEX'] == 1.0])

# print("\n Taux d'hommes ayant un salaire supérieur à 50 000 dollars : ")
# print(taux_homme_sup_50000)

# #Femmes

# femme_et_sup_50000 = features[(features['SEX'] == 2.0) & (features['PINCP'] == True)]
# taux_femme_sup_50000 = len(femme_et_sup_50000) / len(features[features['SEX'] == 2.0])

# print("\n Taux de femmes ayant un salaire supérieur à 50 000 dollars : ")
# print(taux_femme_sup_50000)
# print("QUESTION 2 -----------------------------------------------------------------------------")

# # 4.2 Question 2 ----------------------------------------------------------------------------------

# X_test['PINCP'] = y_test

# # Séparation hommes femmes
# X_test_homme = X_test[X_test['SEX'] == 1.0]
# X_test_femme= X_test[X_test['SEX'] == 2.0]

# y_test_homme = X_test_homme["PINCP"]
# y_test_femme = X_test_femme["PINCP"]

# #on enleve pincp de x_test
# X_test_homme = X_test_homme.drop(columns=['PINCP']) 
# X_test_femme = X_test_femme.drop(columns=['PINCP']) 

# gb_model = joblib.load("/home/koutsodi/5A/apprentisaage-supervise/GradientBoost_BestModel_082.joblib")

# #matrice de confusion femmes
# X_test_scaled_femmes = scaler.transform(X_test_femme) 

# y_predict_femmes = gb_model.predict(X_test_scaled_femmes)
# conf_matrice_femmes = confusion_matrix(y_test_femme, y_predict_femmes) 

# print("Women Confusion Matrix (Gradient Boosting) :\n", conf_matrice_femmes)

# #matrice de confusion femmes
# X_test_scaled_hommes = scaler.transform(X_test_homme) 

# y_predict_hommes = gb_model.predict(X_test_scaled_hommes)
# conf_matrice_hommes = confusion_matrix(y_test_homme, y_predict_hommes) 

# print("Men Confusion Matrix (Gradient Boosting) :\n", conf_matrice_hommes)

# cm_homme = conf_matrice_hommes  
# cm_femme = conf_matrice_femmes  
# TN_h, FP_h, FN_h, TP_h = cm_homme.ravel()
# TN_f, FP_f, FN_f, TP_f = cm_femme.ravel()

# # Statistical Parity
# SP_hommes = (TP_h + FP_h) / (TP_h + FP_h + TN_h + FN_h)
# SP_femmes = (TP_f + FP_f) / (TP_f + FP_f + TN_f + FN_f)

# # Equal Opportunity
# TPR_hommes = TP_h / (TP_h + FN_h) if (TP_h + FN_h) > 0 else 0
# TPR_femmes = TP_f / (TP_f + FN_f) if (TP_f + FN_f) > 0 else 0

# # Predictive Equality
# FPR_hommes = FP_h / (FP_h + TN_h) if (FP_h + TN_h) > 0 else 0
# FPR_femmes = FP_f / (FP_f + TN_f) if (FP_f + TN_f) > 0 else 0


# print("Statistical Parity (hommes): ", SP_hommes)
# print("Statistical Parity (femmes): ", SP_femmes)
# print("Equal Opportunity (TPR hommes): ", TPR_hommes)
# print("Equal Opportunity (TPR femmes): ", TPR_femmes)
# print("Predictive Equality (FPR hommes): ", FPR_hommes)
# print("Predictive Equality (FPR femmes): ", FPR_femmes)


# print("QUESTION 4 -----------------------------------------------------------------------------")

# #4.2 Question 4 ------------------------------------------------------------------------------------------------------------

#suppression de la colonne de genre
# X_train_sans_genre = X_train.drop(columns=['SEX'])
# X_train_sans_genre_scaled = scaler.fit_transform(X_train_sans_genre)


# #on re entraine gradient boosting en enlevant la colonne de genre 
# params_gb = {
#     'n_estimators': [50, 100, 150],         
#     'learning_rate': [0.01, 0.1, 0.5, 1.0], 
#     'max_depth': [3, 5, None]
# }

# gb_model = GradientBoostingClassifier(random_state=42)
# grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=params_gb, cv=5, scoring='accuracy', verbose=10)
# grid_search_gb.fit(X_train_sans_genre_scaled, y_train.values.ravel())
# best_gb_model = grid_search_gb.best_estimator_

# X_test['PINCP'] = y_test

# # Séparation hommes femmes
# X_test_homme = X_test[X_test['SEX'] == 1.0]
# X_test_femme= X_test[X_test['SEX'] == 2.0]

# y_test_homme = X_test_homme["PINCP"]
# y_test_femme = X_test_femme["PINCP"]

# #on enleve pincp de x_test
# X_test_homme = X_test_homme.drop(columns=['PINCP']) 
# X_test_femme = X_test_femme.drop(columns=['PINCP']) 

# #on supprime la colonne de genre pour les tests

# X_test_homme_sans_genre = X_test_homme.drop(columns=['SEX'])
# X_test_femme_sans_genre = X_test_femme.drop(columns=['SEX'])

# #matrice de confusion femmes
# X_test_scaled_femmes = scaler.transform(X_test_femme_sans_genre) 

# y_predict_femmes_sans_genre = best_gb_model.predict(X_test_scaled_femmes)
# conf_matrice_femmes = confusion_matrix(y_test_femme, y_predict_femmes_sans_genre) 

# print("Women Confusion Matrix (Gradient Boosting) without genre column:\n", conf_matrice_femmes)

# #matrice de confusion femmes
# X_test_scaled_hommes = scaler.transform(X_test_homme_sans_genre) 

# y_predict_hommes_sans_genre = best_gb_model.predict(X_test_scaled_hommes)
# conf_matrice_hommes = confusion_matrix(y_test_homme, best_gb_model) 

# print("Men Confusion Matrix (Gradient Boosting) without genre column:\n", conf_matrice_hommes)

# cm_homme = conf_matrice_hommes  
# cm_femme = conf_matrice_femmes  
# TN_h, FP_h, FN_h, TP_h = cm_homme.ravel()
# TN_f, FP_f, FN_f, TP_f = cm_femme.ravel()

# # Statistical Parity
# SP_hommes = (TP_h + FP_h) / (TP_h + FP_h + TN_h + FN_h)
# SP_femmes = (TP_f + FP_f) / (TP_f + FP_f + TN_f + FN_f)

# # Equal Opportunity
# TPR_hommes = TP_h / (TP_h + FN_h) if (TP_h + FN_h) > 0 else 0
# TPR_femmes = TP_f / (TP_f + FN_f) if (TP_f + FN_f) > 0 else 0

# # Predictive Equality
# FPR_hommes = FP_h / (FP_h + TN_h) if (FP_h + TN_h) > 0 else 0
# FPR_femmes = FP_f / (FP_f + TN_f) if (FP_f + TN_f) > 0 else 0


# print("Statistical Parity (hommes): ", SP_hommes)
# print("Statistical Parity (femmes): ", SP_femmes)
# print("Equal Opportunity (TPR hommes): ", TPR_hommes)
# print("Equal Opportunity (TPR femmes): ", TPR_femmes)
# print("Predictive Equality (FPR hommes): ", FPR_hommes)
# print("Predictive Equality (FPR femmes): ", FPR_femmes)
