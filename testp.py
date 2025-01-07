# importing dependencies
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.feature_selection import RFECV


# classical features extraction using PyRadiomics packages
def extracting_classical_features(master_dir_path):
  df = pd.DataFrame()
  img_lablel_list = []
  c = 0
  dir_c = 0

  for dir in os.listdir(master_dir_path):
    for file in os.listdir(os.path.join(f"{master_dir_path}//{dir}")):
      df_img = pd.DataFrame()
      if file.endswith(".jpg"):
        img_path = os.path.join(f"{master_dir_path}//{dir}//{file}")
        img = sitk.ReadImage(img_path)

        # firstorder features extraction module
        from radiomics.firstorder import RadiomicsFirstOrder
        _1stOrder = RadiomicsFirstOrder(img, img)
        _1stOrder.enableAllFeatures()
        computed_features1 = _1stOrder.execute()
        for key, value in computed_features1.items():
          df_img[f"{key}"] = pd.Series(value)

        # shape2D features extraction module
        from radiomics.shape2D import RadiomicsShape2D
        shape2d = RadiomicsShape2D(img, img)
        shape2d.enableAllFeatures()
        computed_features2 = shape2d.execute()
        for key, value in computed_features2.items():
          df_img[f"{key}"] = pd.Series(value)

        # glcm features extraction module
        from radiomics.glcm import RadiomicsGLCM
        RadiomicsGLCM = RadiomicsGLCM(img, img)
        RadiomicsGLCM.enableAllFeatures()  # Enables all first-order features
        computed_features3 = RadiomicsGLCM.execute()
        for key, value in computed_features3.items():
          df_img[f"{key}"] = pd.Series(value)

        # glrlm features extraction module
        from radiomics.glrlm import RadiomicsGLRLM
        RadiomicsGLRLM = RadiomicsGLRLM(img, img)
        RadiomicsGLRLM.enableAllFeatures()
        computed_features4 = RadiomicsGLRLM.execute()
        for key, value in computed_features4.items():
          df_img[f"{key}"] = pd.Series(value)

        # ngtdm features extraction module
        from radiomics.ngtdm import RadiomicsNGTDM
        RadiomicsNGTDM = RadiomicsNGTDM(img, img)
        RadiomicsNGTDM.enableAllFeatures()
        computed_features5 = RadiomicsNGTDM.execute()
        for key, value in computed_features5.items():
          df_img[f"{key}"] = pd.Series(value)

        # gldm features extraction module
        from radiomics.gldm import RadiomicsGLDM
        RadiomicsGLDM = RadiomicsGLDM(img, img)
        RadiomicsGLDM.enableAllFeatures()
        computed_features6 = RadiomicsGLDM.execute()
        for key, value in computed_features6.items():
          df_img[f"{key}"] = pd.Series(value)

        # glszm features extraction module
        from radiomics.glszm import RadiomicsGLSZM
        RadiomicsGLSZM = RadiomicsGLSZM(img, img)
        RadiomicsGLSZM.enableAllFeatures()
        computed_features7 = RadiomicsGLSZM.execute()
        for key, value in computed_features7.items():
          df_img[f"{key}"] = pd.Series(value)

        # image label extraction module
        img_label = file.split(".")[0].split(" ")[0]
        img_lablel_list.append(img_label)
        print(f"image #{c}")
        c+=1

        # concatenating dataframes
        df = pd.concat([df, df_img])


  # image labeling
  df["img_label"] = img_lablel_list

  # dropping unneeded column
  df = df.drop(["Unnamed: 0"], axis=1)
  # csv file saving
  # df.to_csv("extracted_classical_features.csv")
  return df



def data_preprocessing(df):
  pre_scaled = df.iloc[:, :-1].applymap(pd.to_numeric)
  columns_titles = pre_scaled.columns
  scaler = StandardScaler().fit_transform(pre_scaled)
  scaled_data = pd.DataFrame(scaler)
  scaled_data.columns = columns_titles

  label_encoder = LabelEncoder()
  scaled_data['labels']= label_encoder.fit_transform(df.iloc[:,-1])

  return scaled_data


def pearson_correlation_heatmap(df):
  dataplot = sns.heatmap(df.iloc[:,:-1].corr(numeric_only=True), cmap="YlGnBu", annot=False)
  plt.show()

def removing_highly_corr_features(df, show_highly_corr=False, show_curr_corr=False):
  corr_matrix = df.corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

  to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
  processed_df = df.drop(to_drop, axis=1)
  if show_highly_corr == True:
    print("\nThe correlation matrix of the highly correlated features")
    pearson_correlation_heatmap(pd.concat([df[to_drop], df.iloc[:,-1]], axis=1))

  if show_curr_corr == True:
    print("\nThe correlation matrix of the current features")
    pearson_correlation_heatmap(processed_df)

  print(f"# of dropped feaures: {len(to_drop)}")
  print(f"Dropped features are: {to_drop}")
  print(f"# of remained features {processed_df.shape[1]}")
  print(f"Remained features are: {processed_df.columns}")
  return processed_df


# training & evaluating different models [EDITED]
def estimators_training_evaluation(df):

  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]

  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  lr = LogisticRegression(random_state=42, C= 0.1, penalty='l2', solver='liblinear')
  dt = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=6, min_samples_leaf=1, min_samples_split=2)
  rf = RandomForestClassifier(random_state=42, max_depth=6, min_samples_leaf=1, min_samples_split=5)
  knn = KNeighborsClassifier(metric='manhattan', n_neighbors=7, weights='distance')
  svm = SVC(random_state=42, C=6, kernel='rbf')

  estimators = [lr, dt, rf, knn, svm]
  scoring_metric = ['accuracy', 'precision_macro', 'f1_macro']

  accuracy_scores_val = []
  precision_macro_scores_val = []
  f1_macro_scores_val = []
  accuracy_scores_test = []
  precision_macro_scores_test = []
  f1_macro_scores_test = []

  print("## Validation Metrics ##")
  for est in estimators:
    print(f"\n{str(est).split('(')[0]}")
    for metric in scoring_metric:
      scores = cross_val_score(est, x_train, y_train, scoring=metric, cv=10)
      metric_score = np.mean(scores)
      print(f"\t\t>> {metric}: {metric_score*100 :.4f}%")

      if metric == "accuracy":
        accuracy_scores_val.append(metric_score*100)
      elif metric == "precision_macro":
        precision_macro_scores_val.append(metric_score*100)
      elif metric == "f1_macro":
        f1_macro_scores_val.append(metric_score*100)

  print("\n\n## Testing Metrics ##")
  for est in estimators:
    est.fit(x_train, y_train)
    print(f"\n{str(est).split('(')[0]}")
    y_pred = est.predict(x_test)

    accuracy_metric = accuracy_score(y_test, y_pred)
    accuracy_scores_test.append(accuracy_metric*100)
    print(f"\t\t>> accuracy metric: {accuracy_metric*100 :.4f}%")

    precisio_metric = precision_score(y_test, y_pred, average = "macro")
    precision_macro_scores_test.append(precisio_metric*100)
    print(f"\t\t>> precisio metric: {precisio_metric*100 :.4f}%")

    f1_metric = f1_score(y_test, y_pred, average = "macro")
    f1_macro_scores_test.append(f1_metric*100)
    print(f"\t\t>> f1 metric: {f1_metric*100 :.4f}%")

  return accuracy_scores_val, precision_macro_scores_val, f1_macro_scores_val, accuracy_scores_test, precision_macro_scores_test, f1_macro_scores_test

# plotting & saving metric scores figure as png file [DONE]
def estimators_performance_plot(accuracy_scores, precision_macro_scores, f1_macro_scores, save_fig=False):
  estimator = ['lr', 'dt', 'rf', 'knn', 'svm']
  scores = {
      'accuracy scores': accuracy_scores,
      'Precision Scores': precision_macro_scores,
      'F1 Scores': f1_macro_scores}

  x = np.arange(len(estimator))  # the label locations
  width = 0.3  # the width of the bars
  multiplier = 0

  fig, ax = plt.subplots(layout='constrained')

  for attribute, measurement in scores.items():
      offset = width * multiplier
      rects = ax.bar(x + offset, measurement, width, label=attribute)
      ax.bar_label(rects, padding=3, fmt='%.1f')
      multiplier += 1

  ax.set_ylabel("Score")
  ax.set_xlabel("Estimator")
  ax.set_title("Estimator Vs. Metrics Scores")
  ax.set_xticks(x + width, estimator)
  ax.legend(loc='upper center', ncols=3)
  ax.set_ylim(0, 119)
  plt.show()
  if save_fig == True:
    plt.savefig("estimators_performance.png")


features_df = pd.read_csv("full_extracted.csv")
processed_df = data_preprocessing(features_df)

processed_df2 = removing_highly_corr_features(processed_df, show_highly_corr=True, show_curr_corr=True)

accuracy_scores_val, precision_macro_scores_val, f1_macro_scores_val, accuracy_scores_test, precision_macro_scores_test, f1_macro_scores_test = estimators_training_evaluation(processed_df2)






estimators_performance_plot(accuracy_scores_val, precision_macro_scores_val, f1_macro_scores_val, save_fig=True)
estimators_performance_plot(accuracy_scores_test, precision_macro_scores_test, f1_macro_scores_test, save_fig=True)
