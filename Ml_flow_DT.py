import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score

import mlflow
import mlflow.sklearn

import warnings

warnings.filterwarnings('ignore')

def load():
    path = os.getcwd()
    data = pd.read_csv(path+'/train.csv')
    print('Data load completed')
    data = data[:10000]
    X = data.drop(['target','ID_code'], axis=1)
    y = data[['target']]
    return X,y
    
def std(max_depth,min_samples_split):
    mlflow.set_experiment(experiment_name='DecisionTreeModel')
    std = StandardScaler()
    X = load()[0]
    y = load()[1]
    X_std = pd.DataFrame(std.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)
    print("Standarization and Splitting Done")
    with mlflow.start_run():
        dt = DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state=42)
        dt.fit(X_train,y_train)
        y_pred = dt.predict(X_test)
        a_s = accuracy_score(y_test,y_pred)
        p_s = precision_score(y_test,y_pred)
        r_s = recall_score(y_test,y_pred)
        print(a_s,p_s,r_s)
        mlflow.log_param(f'max_depth',max_depth)
        mlflow.log_param(f'min_samples_split',min_samples_split)
        mlflow.log_metric(f'Accuracy',a_s)
        mlflow.log_metric(f'Precision',p_s)
        mlflow.log_metric(f'Recall',r_s)
        # TO register model
        # mlflow.sklearn.log_model(dt, 'Decision Tree', registered_model_name="DecisionTreeModel") 
        mlflow.sklearn.log_model(dt, 'Decision Tree')



if __name__ == '__main__':
    max_depth = int(sys.argv[1])
    min_samples_split = int(sys.argv[2])
    std(max_depth, min_samples_split)