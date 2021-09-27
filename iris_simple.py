import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def iris():
    mlflow.set_experiment(experiment_name='first_mlflow')

    data = load_iris()
    X = pd.DataFrame(data.data, columns = ['Sepal Length','Sepal Width','Petal Length','Petal Width'])
    y = pd.DataFrame(data.target, columns=['Class'])
    print('Data Loaded')

    std = StandardScaler()
    X_std = pd.DataFrame(std.fit_transform(X), columns=X.columns)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

    knn = KNeighborsClassifier()
    knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    model_accuracy = accuracy_score(y_pred, y_test)
    print(model_accuracy)

    mlflow.log_metric('accuracy', model_accuracy)
    mlflow.sklearn.log_model(knn,'model')

if __name__ == '__main__':
    iris()