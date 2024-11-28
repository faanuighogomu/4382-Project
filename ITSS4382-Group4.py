# CODE 1
# import packages
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV

# Load other modules
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# load the data
heart_data = pd.read_csv('Heart Disease.csv')
heart_data = pd.DataFrame(heart_data)
#print(heart_data)

# the input features inlcude all of the columns except for our target variable target
var = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# set all the input variables equal to variable X
X = heart_data[var]

# output (dependent) variable, each record will have a value of either 1 or 0
y = heart_data['target']

# split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

# perform KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

# Check accuracy of the model on test dataset
knn_score = knn.score(X_test, y_test)  
print("KNN Accuracy Score: ", knn_score)


# Grid Search for knn
knn_grid = {'n_neighbors': range(1,11)}
knn_grid_search = GridSearchCV(estimator=knn, param_grid=knn_grid, cv=5)
knn_grid_search.fit(X_train, y_train)
best_knn = knn_grid_search.best_estimator_
print(best_knn)
test_score = best_knn.score(X_test, y_test)
print("The test score is:", {test_score})

predict = knn.predict([[33, 1, 2, 145, 300, 0, 1, 170, 0, 1.0, 2, 0, 3]])
print("Prediction Given Input: ", predict)

healthy = heart_data['target'] == 0
plt.scatter(heart_data['trestbps'], heart_data['chol'], c = healthy)
plt.xlabel('Rest BPS')
plt.ylabel('Cholesterol Level')
plt.title('Rest BPS & Cholesterol Correlation')
plt.legend()
#plt.show()

# logistic regression with GridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression(max_iter=1000)
log_grid = {'C': [0.01, 0.1, 1, 10, 100]}
log_grid_search = GridSearchCV(estimator=logreg, param_grid=log_grid, cv=5, scoring='accuracy')

log_grid_search.fit(X_train, y_train)
best_strength = log_grid_search.best_estimator_
print(best_strength)
log_test_score = best_strength.score(X_test, y_test)
print("The LogReg test score is:", {log_test_score})

# find the coefficients of the model to shows feature strength (aka most important features)
coefficients = abs(best_strength.coef_[0])
feature_names = X_train.columns

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10,10))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color = 'red')

# voting classifier
from sklearn.ensemble import VotingClassifier
vot_clf = VotingClassifier(estimators = [('lr1', logreg), ('knn', knn)], voting='hard')
vot_clf.fit(X_train, y_train)
print("Test score for voting classifier is:" , vot_clf.score(X_test, y_test))


# Naive Bayes (example)
prediction_data = heart_data[(heart_data['age']>=30) & (heart_data['chol']>=300) & (heart_data['trestbps'] >=135) & (heart_data['oldpeak']>=3.0)]
X_prediction = prediction_data[['age', 'trestbps', 'chol', 'oldpeak']]
y_prediction = prediction_data['target']

from sklearn.model_selection import train_test_split
X_prediction, X_test, y_prediction, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import accuracy_score
naiveB = GaussianNB()
naiveB.fit(X_prediction, y_prediction)

y_pred = naiveB.predict(X_prediction)
accurate = accuracy_score(y_prediction, y_pred)
print("This is the accuracy score for the naive bayes model:" ,accurate)

#NOTE: work on reducing the number of columns for more accruacy
# models are all already created, just need to change input features for optimization ^^
# use most_important_features function



# CODE 2
# import packages
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV

# Load other modules
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# load the data
heart_data = pd.read_csv('Heart Disease.csv')
heart_data = pd.DataFrame(heart_data)
#print(heart_data)

# the input features inlcude all of the columns except for our target variable target
var = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# set all the input variables equal to variable X
X = heart_data[var]

# output (dependent) variable, each record will have a value of either 1 or 0
y = heart_data['target']

# split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

# perform KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

# Check accuracy of the model on test dataset
knn_score = knn.score(X_test, y_test)  
print("KNN Accuracy Score: ", knn_score)

# Grid Search for knn
knn_grid = {'n_neighbors': range(1,11)}
knn_grid_search = GridSearchCV(estimator=knn, param_grid=knn_grid, cv=5)
knn_grid_search.fit(X_train, y_train)
best_knn = knn_grid_search.best_estimator_
print(best_knn)
test_score = best_knn.score(X_test, y_test)
print("The test score is:", {test_score})

predict = knn.predict([[33, 1, 2, 145, 300, 0, 1, 170, 0, 1.0, 2, 0, 3]])
print("Prediction Given Input: ", predict)

healthy = heart_data['target'] == 0
plt.scatter(heart_data['trestbps'], heart_data['chol'], c = healthy)
plt.xlabel('Rest BPS')
plt.ylabel('Cholesterol Level')
plt.title('Rest BPS & Cholesterol Correlation')
plt.legend()
#plt.show()

# logistic regression with GridSearch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
log_grid = {'C': [0.01, 0.1, 1, 10, 100]}
log_grid_search = GridSearchCV(estimator=logreg, param_grid=log_grid, cv=5, scoring='accuracy')

# voting classifier
from sklearn.ensemble import VotingClassifier
vot_clf = VotingClassifier(estimators = [('lr1', logreg), ('knn', knn)], voting='hard')
vot_clf.fit(X_train, y_train)
print("Test score for voting classifier is:" , vot_clf.score(X_test, y_test))

# naive bayes example 1
prediction_data = heart_data[(heart_data['age']>=30) & (heart_data['chol']>=300) 
& (heart_data['trestbps'] >=135) & (heart_data['oldpeak']>=3.0)]
X_prediction = prediction_data[['age', 'trestbps', 'chol', 'oldpeak']]
y_prediction = prediction_data['target']

X = heart_data[['age', 'trestbps', 'chol', 'oldpeak']] 
Y = heart_data['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split
(X, y, test_size = 0.20, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
naiveB = GaussianNB()
naiveB.fit(X_train, y_train)

y_pred = naiveB.predict(X_prediction)
accurate = accuracy_score(y_prediction, y_pred)
print("The accuracy score is:", accurate)
for i, (prediction, actual) in enumerate(zip(y_pred, y_prediction)):
    print(f" Prediction data {i+1}: Predicted = {prediction}, actual = {actual}")

# Naive Bayes example 2
pred_data = heart_data[(heart_data['slope']>=2) & (heart_data['sex']==1) 
& (heart_data['exang'] ==1) & (heart_data['cp']>=3)]
X_pred = pred_data[['slope', 'sex', 'exang', 'cp']]
y_pred_2 = pred_data['target']

X = heart_data[['slope', 'sex', 'exang', 'cp']] 
Y = heart_data['target']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
naiveB = GaussianNB()
naiveB.fit(X_train, y_train)

y_pred_again = naiveB.predict(X_pred)
accurate = accuracy_score(y_pred_2, y_pred_again)
print("The accuracy socre is:", accurate)
for i, (prediction, actual) in enumerate(zip(y_pred_again, y_pred_2)):
    print(f" Prediction data {i+1}: Predicted = {prediction}, actual = {actual}")
