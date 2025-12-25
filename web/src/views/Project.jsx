import React from "react";
import Card from "../components/cards/Card";
import CodeCard from "../components/cards/CodeCard";

const Project = () => {
  return (
    <>
      <Card
        mainTitle={`1. Bussiness Understanding`}
        subTitle={`Problem/Opportunity`}
        explain={`
Goal: Detect heart disease earlier to save lives. accelerate diagnosis and improve outcomes for patients.`}
      />
      <Card
        subTitle={`Project Objectives`}
        explain={`
    1. building a model to detect patients with heart-disease and clasify them
    2. cost-benefit advantage over current ways of diagnostic
    3. achieving enough sensitivity for being a useful model to detecting correctly patients
`}
      />
      <Card
        subTitle={`Project Feasibility`}
        explain={`
    1. Small amount of samples may limit our model. In case of that we have to consider augmentation of data.
    2. All data is numeric and thus easy to preprocessing
`}
      />
      <Card
        mainTitle={`2. Data Understanding`}
        subTitle={`Collect Initial Data`}
      />
      <CodeCard
        code={`# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "heart.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "bbyclub/hearth-disease",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())`}
      />
      <CodeCard
        code={`/tmp/ipython-input-3104958777.py:10: DeprecationWarning: Use dataset_load() instead of load_dataset(). load_dataset() will be removed in a future version.
  df = kagglehub.load_dataset(
    
  Downloading from https://www.kaggle.com/api/v1/datasets/download/bbyclub/hearth-disease?dataset_version_number=1&file_name=heart.csv...
  
  100%|██████████| 11.1k/11.1k [00:00<00:00, 1.81MB/s]
  
  First 5 records:    age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
                   0   63    1   3       145   233    1        0      150      0      2.3      0   0     1       1
                   1   37    1   2       130   250    0        1      187      0      3.5      0   0     2       1
                   2   41    0   1       130   204    0        0      172      0      1.4      2   0     2       1
                   3   56    1   1       120   236    0        1      178      0      0.8      2   0     2       1
                   4   57    0   0       120   354    0        1      163      1      0.6      2   0     2       1
  `}
      />
      <Card subTitle={`Describe The Data`} />
      <CodeCard code={`print(df.info())`} />
      <CodeCard
        code={`<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trestbps  303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalach   303 non-null    int64  
 8   exang     303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    int64  
 11  ca        303 non-null    int64  
 12  thal      303 non-null    int64  
 13  target    303 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 33.3 KB
None`}
      />
      <Card
        explain={`    age : Patients Age in years 
        type : Integer
    sex : Male or Female
        type : Boolean
        boolean type : symmetric
    cp : Chest Pain
        type : Integer Ordinal
        value : 0 to 4 (lower to higher pain ratio)
    trestbps : Patients resting blood pressure
        type : Integer
    chol : Cholesterol levels
        type : Integer
    fbs : Fasting blood sugar
        type : Boolean
        boolean type : asymmetric
    restecg : Resting electrocardiographic results
        type : Integer Ordinal
        value : 0 is Normal, 1 for ST-T wave abnormality and 2 for definite left ventricular hypertrophy
    thalach : Maximum heart rate during and exercise
        type : Integer
    exang : Exercise-induced angina
        type : Boolean
        boolean type : asymmetric
    oldpeak : Amount of ST depression on an electrocardiogram induced by exercise relatice to rest.
        type : Float
    slope : ST/HR slope
        type : Integer Ordinal
        value : 0 is Upsloping, 1 Flat and 2 is Downsloping
    ca : Number of major vessels colored by fluoroscopy
        type : Integer Ordinal
        value : Number of colored major vessels
    thal : Thalassemia status
        type : Integer Ordinal
        value : 0 normal, 1 fixed defect, 2 reversible defect
    target : Heart disease
        type : Boolean
        boolean type : asymmetric

there are 303 records of data which will be preprocessed if needed and train a model to detect heart disease.
`}
      />
      <Card
        subTitle={`Explore The Data`}
        explain={`
First we check is there any data missing like null values.`}
      />
      <CodeCard
        code={`summary = df.describe().T
summary['isNull'] = df.isnull().sum()
print(summary)`}
      />
      <CodeCard
        code={`          count        mean        std    min    25%    50%    75%    max  isNull
age       303.0   54.366337   9.082101   29.0   47.5   55.0   61.0   77.0    0 
sex       303.0    0.683168   0.466011    0.0    0.0    1.0    1.0    1.0    0
cp        303.0    0.966997   1.032052    0.0    0.0    1.0    2.0    3.0    0
trestbps  303.0  131.623762  17.538143   94.0  120.0  130.0  140.0  200.0    0
chol      303.0  246.264026  51.830751  126.0  211.0  240.0  274.5  564.0    0
fbs       303.0    0.148515   0.356198    0.0    0.0    0.0    0.0    1.0    0
restecg   303.0    0.528053   0.525860    0.0    0.0    1.0    1.0    2.0    0
thalach   303.0  149.646865  22.905161   71.0  133.5  153.0  166.0  202.0    0
exang     303.0    0.326733   0.469794    0.0    0.0    0.0    1.0    1.0    0
oldpeak   303.0    1.039604   1.161075    0.0    0.0    0.8    1.6    6.2    0
slope     303.0    1.399340   0.616226    0.0    1.0    1.0    2.0    2.0    0
ca        303.0    0.729373   1.022606    0.0    0.0    0.0    1.0    4.0    0
thal      303.0    2.313531   0.612277    0.0    2.0    2.0    3.0    3.0    0
target    303.0    0.544554   0.498835    0.0    0.0    1.0    1.0    1.0    0`}
      />
      <Card
        explain={`As the result above, there is no data with the value null. Calculating the mean, standard deviation, min, max, Q1, Q2, Q3 for each entry.`}
      />
      <CodeCard
        code={`import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 14))
for i, feature in enumerate(df.columns[:-1]):
  plt.subplot(4, 4, i+1)
  if df[feature].dtype == 'float64':
    sns.histplot(df[feature], kde=True, bins=20, color='mediumseagreen')
  else:
    sns.countplot(x=feature, data=df, palette='Set2')
  plt.title(feature)
  plt.tight_layout()
plt.show()`}
      />
      <img
        className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl w-full justify-center bg-neutral-950 shadow"
        src="/public/countplot.png"
      />
      <Card
        explain={`The images display us each variable's distribution and count. We can see that some variables are imbalanced and that several distrubutions are skewed.`}
      />
      <CodeCard
        code={`plt.figure(figsize=(14, 14))
for i, feature in enumerate(df.columns[:-1]):
  plt.subplot(4, 4, i+1)
  sns.boxplot(x='target', y=feature, data=df, palette='Set2')
  plt.title(f'{feature} by target')
  plt.tight_layout()
plt.show()`}
      />
      <img
        className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl w-full justify-center bg-neutral-950 shadow"
        src="/public/boxplot.png"
      />
      <Card
        explain={`From the image we can observe the data, decide which variables to preprocess, and prioritize features during models building.`}
      />
      <CodeCard
        code={`feature = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
sns.pairplot(df[feature + ['target']], hue='target', palette='Set2')
plt.show()`}
      />
      <img
        className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl w-full justify-center bg-neutral-950 shadow"
        src="/public/pairplot.png"
      />
      <Card
        explain={`In the above image we wanted to compare the features together so we have an understanding when features create distinct clusters.`}
      />
      <CodeCard
        code={`correlation = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, annot=True, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()`}
      />
      <img
        className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl w-full justify-center bg-neutral-950 shadow"
        src="/public/correlation.png"
      />
      <Card
        explain={`After calculating the correlation between features we used a heatmap to visualize it. So now we know which features increase together and which decrease with increase of the other. From the above heatmap we obeserve that non of the features are strongly correlated, thus there is no need for feature reduction.`}
      />
      <CodeCard
        code={`counts = df['target'].value_counts().sort_index()

plt.pie(counts.values, labels=counts.index.astype(str), startangle=90, autopct='%1.f%%', colors=['#72b6a1','#e99675'])
plt.axis('equal')
plt.title('Target distribution')
plt.show()`}
      />
      <img
        className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl w-full justify-center bg-neutral-950 shadow"
        src="/public/distribution.png"
      />
      <Card
        explain={`There is no need to use methods like "Smote" for generating data objects for balancing the dataset outcome.`}
      />
      <Card subTitle={`Verify Data Quality`} />
      <CodeCard
        code={`verify = {}

verify['isDuplicated'] = df.duplicated().sum()

if verify['isDuplicated'] > 0:
  df.drop_duplicates(inplace=True)
  verify['isDuplicated'] = df.duplicated().sum()
verify['sex'] = df['sex'].unique()
verify['cp'] = df['cp'].unique()
verify['fbs'] = df['fbs'].unique()
verify['restecg'] = df['restecg'].unique()
verify['exang'] = df['exang'].unique()
verify['slope'] = df['slope'].unique()
verify['ca'] = df['ca'].unique()
verify['thal'] = df['thal'].unique()

print(f"duplicate amout : {verify['isDuplicated']} ,\nunique values for sex: {verify['sex']},\ncp: {verify['cp']},\nfbs: {verify['fbs']},\nrestecg: {verify['restecg']},\nexang: {verify['exang']},\nslope: {verify['slope']},\nca: {verify['ca']},\nthal: {verify['thal']}")`}
      />
      <CodeCard
        code={`duplicate amout : 0 ,
unique values for sex: [1 0],
cp: [3 2 1 0],
fbs: [1 0],
restecg: [0 1 2],
exang: [0 1],
slope: [0 2 1],
ca: [0 2 1 3 4],
thal: [1 2 3 0]`}
      />
      <Card
        explain={`We check for the values with ordinal or boolean value so we don't have an inconsistency in values. Also checked data types above and results are all Integer or Float so data is consistent across the records.`}
      />
      <Card
        mainTitle={`3. Data Preparation`}
        subTitle={`Data Cleaning`}
        explain={`
There is no missing values but we should detect and remove the outliers from our data. In order to achive this data first we'll define the belowe function to mask outlier data.`}
      />
      <CodeCard
        code={`def iqr(feature):
  Q1 = feature.quantile(0.25)
  Q3 = feature.quantile(0.75)
  iqr = Q3 - Q1
  LBound = Q1 - 1.5 * iqr
  UBound = Q3 + 1.5 * iqr
  return (feature >= LBound) & (feature <= UBound)`}
      />
      <Card
        explain={`From the second part of this project and boxplots we realized some of the data are outliers. These outliers are visible in the features listed.
    - age
    - trestbps
    - chol
    - thalach
    - oldpeak
`}
      />
      <CodeCard
        code={`df_clean = df.copy()
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for feature in features:
  mask = iqr(df_clean[feature])
  df_clean = df_clean[mask].reset_index(drop=True)
  
print(df_clean.shape)`}
      />
      <CodeCard code={`(283, 14)`} />
      <Card
        explain={`As seen the records has been decreased and most of the outliers been removed from the dataset.`}
      />
      <Card
        subTitle={`Data Transformation`}
        explain={`
Some of our numeric values are useful if we normalize them. We are going to use Z-Score for normalization.`}
      />
      <CodeCard
        code={`from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_Z_Score = df_clean.copy()
df_Z_Score[features] = scaler.fit_transform(df_clean[features])

print(df_Z_Score)`}
      />
      <CodeCard
        code={`          age  sex  cp  trestbps      chol  fbs  restecg   thalach  exang  oldpeak  slope  ca  thal  target  
0    0.982275    1   3  0.984117 -0.208961    1        0  0.001716     0  1.315709      0   0     1       1   
1   -1.871092    1   2  0.007591  0.172417    0        1  1.635162     0  2.486915      0   0     2       1  
2   -1.432112    0   1  0.007591 -0.859547    0        0  0.972954     0  0.437305      2   0     2       1  
3    0.214061    1   1 -0.643426 -0.141659    0        1  1.237837     0 -0.148297      2   0     2       1  
4    0.323805    0   0 -0.643426  2.505551    0        1  0.575629     1 -0.343498      2   0     2       1   
..        ...  ...  ..       ...       ...  ...      ...       ...   ...       ...    ...  ..    ..      ..
278  0.323805    0   0  0.658608 -0.029489    0        1 -1.190258     1 -0.733900      1   0     3       0   
279 -0.993133    1   3 -1.294442  0.486493    0        1 -0.792933     0  0.242104      1   0     3       0  
280  1.530999    1   0  0.919015 -1.106321    1        1 -0.395609     0  2.389314      1   2     3       0  
281  0.323805    1   0  0.007591 -2.497228    0        1 -1.543436     1  0.242104      1   1     3       0  
282  0.323805    0   1  0.007591 -0.141659    0        0  1.061249     0 -0.929101      1   1     2       0  
[283 rows x 14 columns]`}
      />
      <CodeCard
        code={`import joblib

joblib.dump(scaler, 'zscore_scaler.joblib')`}
      />
      <Card
        subTitle={`Partition Data`}
        explain={`
First we split data into train and test.`}
      />
      <CodeCard
        code={`from sklearn.model_selection import train_test_split

X = df_Z_Score.drop(columns='target')
Y = df_Z_Score['target']

X_train, X_test, Y_train, Y_test = train_test_split( X, Y,test_size=0.30,random_state=42)`}
      />
      <Card
        mainTitle={`4. Modeling`}
        explain={`
(Parameter selection by GridsearchCV then Train and save trained models)`}
      />
      <Card subTitle={`KNN`} />
      <CodeCard
        code={`from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()

knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "p": [1, 2]  # p=1 -> Manhattan, p=2 -> Euclidean
}`}
      />
      <CodeCard
        code={`from sklearn.model_selection import GridSearchCV

knn_gs = GridSearchCV(
    estimator=knn_model,
    param_grid=knn_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

knn_gs.fit(X_train, Y_train)

print("Best KNN params:", knn_gs.best_params_)
print("Best KNN score:", knn_gs.best_score_)
knn_best = knn_gs.best_estimator_`}
      />
      <CodeCard
        code={`Best KNN params: {'n_neighbors': 7, 'p': 2, 'weights': 'uniform'}
Best KNN score: 0.8285897435897436`}
      />
      <CodeCard
        code={`import joblib

joblib.dump(knn_best, "model_knn.pkl")
print("KNN model saved as model_knn.pkl")`}
      />
      <Card subTitle={`SVM`} />
      <CodeCard
        code={`from sklearn.svm import SVC

svm_model = SVC()

svm_param_grid = {
    "C": [0.1, 1, 10, 50, 100],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"]
}`}
      />
      <CodeCard
        code={`from sklearn.model_selection import GridSearchCV

svm_gs = GridSearchCV(
    estimator=svm_model,
    param_grid=svm_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

svm_gs.fit(X_train, Y_train)

print("Best SVM params:", svm_gs.best_params_)
print("Best SVM score:", svm_gs.best_score_)

svm_best = svm_gs.best_estimator_`}
      />
      <CodeCard
        code={`Best SVM params: {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
Best SVM score: 0.8433333333333332`}
      />
      <CodeCard
        code={`import joblib

joblib.dump(svm_best, "model_svm.pkl")
print("SVM model saved as model_svm.pkl")`}
      />
      <Card subTitle={`Decision Tree`} />
      <CodeCard
        code={`from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)

dt_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}`}
      />
      <CodeCard
        code={`from sklearn.model_selection import GridSearchCV

dt_gs = GridSearchCV(
    estimator=dt_model,
    param_grid=dt_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

dt_gs.fit(X_train, Y_train)
print("Best Decision Tree params:", dt_gs.best_params_)
print("Best Decision Tree score:", dt_gs.best_score_)
dt_best = dt_gs.best_estimator_`}
      />
      <CodeCard
        code={`Best Decision Tree params: {'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best Decision Tree score: 0.7825641025641026`}
      />
      <CodeCard
        code={`import joblib

joblib.dump(dt_best, "model_decision_tree.pkl")
print("Decision Tree model saved as model_decision_tree.pkl")`}
      />
      <CodeCard
        code={`from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(dt_best,
          feature_names=X_train.columns,
          class_names=["No Disease", "Disease"],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()`}
      />{" "}
      <img
        className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl w-full justify-center bg-neutral-950 shadow"
        src="/public/decisiontree.png"
      />
      <Card subTitle={`Random Forest`} />
      <CodeCard
        code={`from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

rf_param_grid = {
    "n_estimators": [25, 50, 100],
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}`}
      />
      <CodeCard
        code={`from sklearn.model_selection import GridSearchCV

rf_gs = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_gs.fit(X_train, Y_train)

print("Best Random Forest params:", rf_gs.best_params_)
print("Best Random Forest score:", rf_gs.best_score_)

rf_best = rf_gs.best_estimator_`}
      />
      <CodeCard
        code={`Best Random Forest params: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
Best Random Forest score: 0.8333333333333334`}
      />
      <CodeCard
        code={`import joblib

joblib.dump(rf_best, "model_random_forest.pkl")
print("Random Forest model saved as model_random_forest.pkl")`}
      />
      <Card subTitle={`XGBoost`} />
      <CodeCard
        code={`from xgboost import XGBClassifier

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

xgb_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7,10,15],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0]
}`}
      />
      <CodeCard
        code={`from sklearn.model_selection import GridSearchCV

xgb_gs = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

xgb_gs.fit(X_train, Y_train)

print("Best XGBoost params:", xgb_gs.best_params_)
print("Best XGBoost score:", xgb_gs.best_score_)

xgb_best = xgb_gs.best_estimator_
`}
      />
      <CodeCard
        code={`Best XGBoost params: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 0.8}
Best XGBoost score: 0.8330769230769229`}
      />
      <CodeCard
        code={`import joblib

joblib.dump(xgb_best, "model_xgboost.pkl")
print("XGBoost model saved as model_xgboost.pkl")
`}
      />
      <Card subTitle={`G-Naive-Bayes`} />
      <CodeCard
        code={`from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
`}
      />
      <CodeCard
        code={`nb_model.fit(X_train, Y_train)

import joblib
joblib.dump(nb_model, "model_naive_bayes.pkl")
print("Naive Bayes model saved as model_naive_bayes.pkl")`}
      />
      <Card subTitle={`MLP`} />
      <CodeCard
        code={`from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(max_iter=500, random_state=42)

mlp_param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (50,100), (50,150,75)],
    "activation": ["relu", "tanh", "logistic"],
    "solver": ["adam", "sgd"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"]
}
`}
      />
      <CodeCard
        code={`from sklearn.model_selection import GridSearchCV

mlp_gs = GridSearchCV(
    estimator=mlp_model,
    param_grid=mlp_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

mlp_gs.fit(X_train, Y_train)

print("Best MLP params:", mlp_gs.best_params_)
print("Best MLP score:", mlp_gs.best_score_)

mlp_best = mlp_gs.best_estimator_
`}
      />
      <CodeCard
        code={`Best MLP params: {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'solver': 'adam'}
Best MLP score: 0.8435897435897436`}
      />
      <CodeCard
        code={`import joblib

joblib.dump(mlp_best, "model_mlp.pkl")
print("MLP model saved as model_mlp.pkl")
`}
      />
      <Card subTitle={`Evaluate Model Performance`} />
      <CodeCard
        code={`import os

model_files = [
    "model_knn.pkl",
    "model_svm.pkl",
    "model_decision_tree.pkl",
    "model_random_forest.pkl",
    "model_xgboost.pkl",
    "model_naive_bayes.pkl",
    "model_mlp.pkl"
]
import pandas as pd
df_models = pd.DataFrame({
    "Model Name": ["KNN", "SVM", "Decision Tree", "Random Forest", "XGBoost", "Naive Bayes","MLP"],
    "File Name": model_files,
    "File Path": [os.path.abspath(f) for f in model_files]
})
df_models`}
      />
      <CodeCard
        code={`       Model Name        File Name                   File Path
0 	   KNN 	             model_knn.pkl 	             /content/model_knn.pkl
1      SVM 	             model_svm.pkl 	             /content/model_svm.pkl
2      Decision Tree 	 model_decision_tree.pkl 	 /content/model_decision_tree.pkl
3 	   Random Forest 	 model_random_forest.pkl 	 /content/model_random_forest.pkl
4 	   XGBoost 	         model_xgboost.pkl 	         /content/model_xgboost.pkl
5 	   Naive Bayes 	     model_naive_bayes.pkl 	     /content/model_naive_bayes.pkl
6 	   MLP 	             model_mlp.pkl 	             /content/model_mlp.pkl`}
      />
      <Card subTitle={`Evaluation`} />
      <CodeCard
        code={`import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
models = {
    "KNN": "model_knn.pkl",
    "SVM": "model_svm.pkl",
    "Decision Tree": "model_decision_tree.pkl",
    "Random Forest": "model_random_forest.pkl",
    "XGBoost": "model_xgboost.pkl",
    "Naive Bayes": "model_naive_bayes.pkl",
    "MLP": "model_mlp.pkl"
}
results = []
for name, file in models.items():
    model = joblib.load(file)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        from sklearn.preprocessing import LabelBinarizer
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(Y_test, y_pred),
        "Recall": recall_score(Y_test, y_pred),
        "Precision": precision_score(Y_test, y_pred),
        "F1-score": f1_score(Y_test, y_pred),
        "AUC": roc_auc_score(Y_test, y_prob)
    })
import pandas as pd
df_results = pd.DataFrame(results)
df_results`}
      />
      <CodeCard
        code={`               Model            Accuracy    Recall      Precision 	F1-score 	AUC
0              KNN              0.788235 	0.833333 	0.800000 	0.816327 	0.844876
1 	           SVM 	            0.823529 	0.937500 	0.789474 	0.857143 	0.896396
2 	           Decision Tree 	0.694118 	0.666667 	0.761905 	0.711111 	0.698198
3 	           Random Forest 	0.823529 	0.833333 	0.851064 	0.842105 	0.908784
4 	           XGBoost 	        0.764706 	0.750000 	0.818182 	0.782609 	0.862050
5 	           Naive Bayes 	    0.800000 	0.833333 	0.816327 	0.824742 	0.894707
6 	           MLP 	            0.823529 	0.895833 	0.811321 	0.851485 	0.896959`}
      />
      <Card subTitle={`ROC Curve + Individual Confusion Matrix`} />
      <CodeCard
        code={`import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

plt.figure(figsize=(8,6))
for name, file in models.items():
    model = joblib.load(file)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    fpr, tpr, _ = roc_curve(Y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

for name, file in models.items():
    model = joblib.load(file)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(Y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {name}")
    plt.show()`}
      />
      <img
        className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl w-full justify-center bg-neutral-950 shadow"
        src="/public/roccurve.png"
      />
      <img
        className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl w-full justify-center bg-neutral-950 shadow"
        src="/public/confusionmatrix.png"
      />
    </>
  );
};

export default Project;
