from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

iris_df = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_df.data, iris_df.target,
                                                    test_size=0.3, random_state=0)

## Pipelines Creation
## 1. Data Preprocessing by using Standard Scaler
## 2. Reduce Dimension using PCA
## 3. Apply Classifier

pipeline_lr = Pipeline([('scalar1', StandardScaler()),
                        ('pca1', PCA(n_components=2)),
                         ('lr_classifier', LogisticRegression(
                             random_state=0))])

pipeline_dt = Pipeline([('scalar2', StandardScaler()),
                        ('pca2', PCA(n_components=2)),
                        ('dt_classifier', DecisionTreeClassifier())])

pipeline_randomforest = Pipeline([('scalar3', StandardScaler()),
                                  ('pca3', PCA(n_components=2)),
                                  ('rf_classifier', RandomForestClassifier())])


## Lets make the list of pipelines
pipelines = [pipeline_lr, pipeline_dt, pipeline_randomforest]

best_accuracy = 0.0
best_classifier = 0
best_pipeline = ""

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict =  {0 : 'Logistic Regression', 1 : 'Decision Tree',
              2 : 'RandomForest'}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

for i, model in enumerate(pipelines):
    print("{} Test Accuracy : {}".format(pipe_dict[i],
                                         model.score(X_test, y_test)))
    
