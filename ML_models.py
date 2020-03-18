#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc

#Read the data
df = pd.read_csv('bs140513_032310.csv')
print(df.nunique())

#Drop unnecessary columns
df = df.drop(['zipcodeOri', 'zipMerchant', 'step', 'customer'], axis = 1)


#Clean the data
df['age'] = df['age'].apply(lambda x: x[1]).replace('U', 7).astype(int)
df['gender'] = df['gender'].apply(lambda x: x[1])
df['merchant'] = df['merchant'].apply(lambda x: x[1:-1])
df['category'] = df['category'].apply(lambda x: x[1:-1])

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['amount', 'fraud']] = scaler.fit_transform(df[['amount', 'fraud']])
 
#Describe the data as features and label
#Set the feature and class labels
features = df.drop('fraud', axis = 1)
label = df.fraud

#One hot encoding
features = pd.get_dummies(features)

#Splitting data into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, label, train_size = 0.8, random_state = 0)


from sklearn import metrics

"""Logistic Regression""" 
print("Logistic Regression")
#from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 1000, class_weight = {1: 0.7, 0: 0.3}, penalty = 'l1', solver = 'liblinear')
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
#Calculating accuracy by confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score: ", acc)
#Creating a classification report
cr = classification_report(y_test, y_pred)
print(cr)

#Use GridSearchCV for better estimation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']
param_grid = dict(penalty=penalty,
                  C=C,
                  class_weight=class_weight,
                  solver=solver)

grid = GridSearchCV(estimator=lr,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    verbose=1,
                    n_jobs=-1)
grid_result = grid.fit(x_train, y_train)
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)




"""SGD Classifier"""
print("SGD Classifier")
#from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
#Calculating accuracy by confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score: ", acc)
#Creating a classification report
cr = classification_report(y_test, y_pred)
print(cr)



"""Decision Tree Classifier"""
print("Decision tree Classifier")
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
dtc.fit(x_train, y_train)
#Predict for x_test
y_pred = dtc.predict(x_test)
#Checking accuracy of Naive Bayes
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
acc = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",acc)
#Creating a classification report
cr = classification_report(y_test, y_pred)
print(cr)


"""KNN"""
print("K nearest Neighbor")
#Fitting KNN classifier with L2 norm (Euclideon Distance)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
# Creating odd list K for KNN
neighbors = list(range(1,50,2))
# empty list that will hold cv scores
cv_scores = [ ]
#perform 10-fold cross-validation
for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors = K)
    scores = cross_val_score(knn,x_train,y_train,cv = 10,scoring = "accuracy")
    cv_scores.append(scores.mean())
# Changing to mis classification error
mse = [1-x for x in cv_scores]
# determing best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal no. of neighbors is {}".format(optimal_k))
def plot_accuracy(knn_list_scores):
    pd.DataFrame({"K":[i for i in range(1,50,2)], "Accuracy":knn_list_scores}).set_index("K").plot.bar(figsize= (9,6),ylim=(0.78,0.83),rot=0)
    plt.show()
plot_accuracy(cv_scores)
knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
knn.fit(x_train, y_train)
#Make prediction
y_pred = knn.predict(x_test)
#Calculating accuracy
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score: ", acc)
#Creating a classification report
cr = classification_report(y_test, y_pred)
print(cr)


"""Naive Bayes"""
print("Naive Bayes")
#Fit the model to Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
#Predict values for x_test
y_pred = nb.predict(x_test)
#Checking accuracy of Naive Bayes
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
acc = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",acc)
#Creating a classification report
cr = classification_report(y_test, y_pred)
print(cr)

"""Random Forest"""
print("Random Forrest")
#Fit the Random Forrest Classification Model
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import validation_curve

#Check for AUC curve
n_estimators = []
for i in range(1, 40):
    n_estimators.append(i) 
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(x_train, y_train)
   train_pred = rf.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()
#Best estimator = 15

#Check for max_depth
max_depths = np.linspace(1, 50, 50, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
   rf.fit(x_train, y_train)
   train_pred = rf.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()
#Max depth = 19

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 15, max_depth = 19, criterion = "entropy", random_state = 0, min_samples_split = 20)
rfc.fit(x_train, y_train)
#Predict for x_test
y_pred = rfc.predict(x_test)
#Checking accuracy of Naive Bayes
cm = metrics.confusion_matrix(y_test, y_pred) 
print(cm)
acc = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy score:",acc)
#Creating a classification report
cr = classification_report(y_test, y_pred)
print(cr)

'''
from scipy.stats import randint as sp_randint
# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None], "max_features": sp_randint(1, x_train.shape[1]), "min_samples_split": sp_randint(2, 11), "bootstrap": [True, False], "n_estimators": sp_randint(100, 500)}
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=10, cv=5, iid=False, random_state=42)
random_search.fit(x_train, y_train)
print(random_search.best_params_)
'''

"""SVC Linear"""
print("SVM with linear kernel")
from sklearn.svm import SVC
svr = SVC(kernel = "linear", random_state = 0)
svr.fit(x_train, y_train)
#Making predictions
y_pred = svr.predict(x_test)
#Checking Accuracy for x_test
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
#Creating a classification report
cr = classification_report(y_test, y_pred)
print(cr)


"""SVC Radial"""
print("SVM with Radial function kernel")
#Fitting SVM model with Radial Basis Function kernel
from sklearn.svm import SVC
svr = SVC(kernel = "rbf", random_state = 0)
svr.fit(x_train, y_train)
#Making predictions
y_pred = svr.predict(x_test)
#Checking Accuracy for x_test
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ", cm)
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
#Creating a classification report
cr = classification_report(y_test, y_pred)
print(cr)

