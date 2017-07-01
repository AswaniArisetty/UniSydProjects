"""
COMP 5318 - Machine Learning and Data Mining 
Assignment 2 - Comparison of Classifiers - Forest Cover Type
470162451_460283346_470148514
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from scipy import interp
from itertools import cycle


# Plot learning curve functionality adapted from examples provided in sklearn
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# plot ROC curve functionality adapted from examples provided in sklearn
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_ROC_curve(title_plot,y_test_bin,y_score_bin):
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

    colors = cycle(['red','blue','green','yellow','aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
		label='ROC curve of class {0} (area = {1:0.2f})'
		''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_plot)
    plt.legend(loc="lower right")
    plt.show()

# Create column names for dataset.
df_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
              'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
              'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2',
              'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4',
              'Soil_Type_5',
              'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11',
              'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15',
              'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21',
              'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25',
              'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31',
              'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35',
              'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39', 'Soil_Type_40', 'Cover_Type']

# Read the csv file.
cover_type_df = pd.read_csv('covtype.csv', header=None, index_col=False)
cover_type_df.columns = df_columns

# Box Plot of different attributes to cover type column
bp_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
              'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
              'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2',
              'Wilderness_Area3', 'Wilderness_Area4', 'Cover_Type']
x = bp_columns[len(bp_columns) - 1]
y = bp_columns[0:len(bp_columns) - 1]
for i in range(0, len(bp_columns) - 1):
    sns.boxplot(data=cover_type_df, x=x, y=y[i])
    plt.show()
# Seperate the input dataset and Output dataset
X = cover_type_df.iloc[:, :54]
Y = cover_type_df['Cover_Type']

# Train test split - 33% Testing data and 66% Training data
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Pearsons Correlation Heat Map
fig1 = plt.figure(figsize=(10, 10))
foo = sns.heatmap(cover_type_df.iloc[:, :14].corr(), cbar=False, vmax=0.6, square=True, annot=True)
fig1.savefig("Heatmap- Forest Cover.pdf")
plt.show()

# Scaling the numeric variables - First 10 columns using Standard and Minmax scaler
X_SS = np.concatenate((StandardScaler().fit_transform(X_train.iloc[:, 0:10]), X_train.iloc[:, 10:]), axis=1)
X_test_SS = np.concatenate((StandardScaler().fit_transform(X_test.iloc[:, 0:10]), X_test.iloc[:, 10:]), axis=1)

X_MM = np.concatenate((MinMaxScaler().fit_transform(X_train.iloc[:, 0:10]), X_train.iloc[:, 10:]), axis=1)
X_test_MM = np.concatenate((MinMaxScaler().fit_transform(X_test.iloc[:, 0:10]), X_test.iloc[:, 10:]), axis=1)

##############################################################################################
#####                                  Logistic Regression                             #######
##############################################################################################

# Grid Search for Logistic regression. Intentionally commented as the run is time consuming
"""
# GridSearch for Logistic Regression
sample_C = [0.01,0.1,1,10]
param_grid = dict(C=sample_C)
model = LogisticRegression()
grid = GridSearchCV(estimator=model, param_grid=param_grid,verbose=1, scoring="f1_macro")
grid.fit(X_SS, Y_train)
print(grid.best_score_)
print(grid.best_estimator_.C)
"""

# Using Standard Scalar transformation on input continuous features gave a good accuracy.
# Using Grid Search CV it is found that C = 10 provided the best accuracy and F1 score.
start_time = time.time()
kfold = KFold(n_splits=10, random_state=7)
modelLR = LogisticRegression(C=10)
resultsLR = cross_val_score(modelLR, X_SS, Y_train, cv=kfold)
modelLR.fit(X_SS, Y_train)
result = modelLR.score(X_test_SS, Y_test)
print("Test Dataset: Accuracy Logisitic Regression :", result * 100.0)
predicted_LR = modelLR.predict(X_test_SS)
print("Logistic Regression: Classification Report")
reportLR = classification_report(Y_test, predicted_LR)
print(reportLR)

conf_LR = confusion_matrix(Y_test, predicted_LR)
print(conf_LR)

output_type = ['Spruce/Fir', 'Lodgepole', 'Ponderosa', 'Cottonwood', 'Aspen', 'Douglas', 'Krummholz']
df_cm_LR = pd.DataFrame(conf_LR, index=[i for i in output_type], columns=[i for i in output_type])
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm_LR, annot=True)
plt.show()

classifierLR = OneVsRestClassifier(LogisticRegression(C=10, random_state=7))
y_score_LR = classifierLR.fit(X_SS, Y_train).predict(X_test_SS)

y_train_bin = label_binarize(Y_train, classes=[1, 2, 3, 4, 5, 6, 7])
y_test_bin = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6, 7])
y_score_LR_bin = label_binarize(y_score_LR, classes=[1, 2, 3, 4, 5, 6, 7])

n_classes = y_train_bin.shape[1]

# ROC plot for Logistic Regression
plot_ROC_curve('ROC curve for Logistic Regression Classifier',y_test_bin,y_score_LR_bin)

# Learning Curve -  Logistic Regression
title = "Learning Curves (Logistic Regression)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=7)
estimator = LogisticRegression(C=10)
pltlr = plot_learning_curve(estimator, title, np.array(X), np.array(Y), cv=cv)
pltlr.show()
##############################################################################################
#####                                  k-Neighbors Classifier                          #######
##############################################################################################

kfold = KFold(n_splits=10, random_state=7)
modelknn = KNeighborsClassifier(n_neighbors=1)
resultsknn = cross_val_score(modelknn, X_train, Y_train, cv=kfold)
modelknn.fit(X_train, Y_train)
result = modelknn.score(X_test, Y_test)
print("Test Dataset: Accuracy K nearest neighbours :", result * 100.0)
predicted_knn = modelknn.predict(X_test)
print("K nearest neighbours: Classification Report")
reportknn = classification_report(Y_test, predicted_knn)
print(reportknn)

conf_knn = confusion_matrix(Y_test, predicted_knn)
print(conf_knn)

output_type = ['Spruce/Fir', 'Lodgepole', 'Ponderosa', 'Cottonwood', 'Aspen', 'Douglas', 'Krummholz']
df_cm_knn = pd.DataFrame(conf_knn, index=[i for i in output_type], columns=[i for i in output_type])
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm_knn, annot=True)
plt.show()

classifierknn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=1))
y_score_knn = classifierknn.fit(X_train, Y_train).predict(X_test)

y_score_knn_bin = label_binarize(y_score_knn, classes=[1, 2, 3, 4, 5, 6, 7])
plot_ROC_curve('ROC curve for k-Nearest Neighbor Classifier',y_test_bin,y_score_knn_bin)

# Learning Curve -  Knn
title = "Learning Curves (KNN Classifier)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=7)
estimator = KNeighborsClassifier(n_neighbors=1)
pltknn = plot_learning_curve(estimator, title, np.array(X), np.array(Y), cv=cv)
pltknn.show()
##############################################################################################
#####                                  Random Forest                                   #######
##############################################################################################
# Intentionally commented as the run for grid search is time consuming
# num_trees = 200 and min_samples_leaf=1 are the optimal hyperparameters after tuning
"""
# Random Forest - Grid Search CV for n_estimators
sample_n_estimators = [100,200,300,500]
param_grid = dict(n_estimators = sample_n_estimators)
model = RandomForestClassifier(max_features="auto")
grid = GridSearchCV(estimator=model, param_grid=param_grid,verbose=1, scoring="f1_macro")
grid.fit(X_MinMax, Y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_estimators)
"""

"""
# Random Forest - Grid Search CV for min_samples_leaf
sample_leaf_options = [1,5,10,50,100,200,300,500]
param_grid = dict(min_samples_leaf=sample_leaf_options)
model = RandomForestClassifier(n_estimators=200,max_features="auto")
grid = GridSearchCV(estimator=model, param_grid=param_grid,verbose=1, scoring="f1_macro")
grid.fit(X_, Y_train)
print(grid.best_score_)
print(grid.best_estimator_.min_samples_leaf)
"""

# Random Forest classifier with input preprocessed using MinMax Scaler was shown to give the best accuracy.
# Used grid search evaluation to tune 2 parameteres - n_estimators and min_samples_leaf. n_estimators - 200
# and min_samples_leaf = 1 provides the best F1 score.

num_trees = 200
kfold = KFold(n_splits=10, random_state=7)
modelRFR = RandomForestClassifier(n_estimators=num_trees, max_features="auto", min_samples_leaf=1)
resultsRFR = cross_val_score(modelRFR, X_MM, Y_train, cv=kfold)
modelRFR.fit(X_MM, Y_train)
result = modelRFR.score(X_test_MM, Y_test)
print("Test Dataset: Accuracy Random Forest Classifier :", result * 100.0)
predicted_RF = modelRFR.predict(X_test_MM)
print("Random Forest Classifier: Classification Report")
reportRF = classification_report(Y_test, predicted_RF)
print(reportRF)
print("Feature Importance")
print(modelRFR.feature_importances_)

conf_RF = confusion_matrix(Y_test, predicted_RF)
print(conf_RF)

output_type = ['Spruce/Fir', 'Lodgepole', 'Ponderosa', 'Cottonwood', 'Aspen', 'Douglas', 'Krummholz']
df_cm_RF = pd.DataFrame(conf_RF, index=[i for i in output_type], columns=[i for i in output_type])
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm_RF, annot=True)
plt.show()

# Feature Importance of Random Forest
fig1 = plt.figure(figsize=(8, 14))
importances = modelRFR.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train.columns)
plt.xlabel('Relative Importance')
plt.show()

classifierRF = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=200, max_features="auto", min_samples_leaf=1, random_state=7))
y_score_RF = classifierRF.fit(X_MM, Y_train).predict(X_test_MM)

y_score_RF_bin = label_binarize(y_score_RF, classes=[1, 2, 3, 4, 5, 6, 7])

plot_ROC_curve('ROC curve for Random Forest Classifier',y_test_bin,y_score_RF_bin)

##############################################################################################
#####                                  Gradient Boosting Machine                       #######
##############################################################################################

# Used same tuning parameters as Random forest for n_estimators and min_samples_leaf
num_trees = 200
kfold = KFold(n_splits=10, random_state=7)
modelGBC = GradientBoostingClassifier(n_estimators=num_trees, max_features="auto", min_samples_leaf=1)
resultsGBC = cross_val_score(modelGBC, X_MM, Y_train, cv=kfold)
modelGBC.fit(X_MM, Y_train)
modelGBC.fit(X_MM, Y_train)
result = modelGBC.score(X_test_MM, Y_test)
print("Test Dataset: Accuracy Gradient Boosting Classifier :", result * 100.0)
predicted_GBC = modelGBC.predict(X_test_MM)
print("Gradient Boosting Classifier: Classification Report")
reportGB = classification_report(Y_test, predicted_GBC)
print(reportGB)
print("Feature Importance")
print(modelGBC.feature_importances_)
print("Gradient Boosting Classifier: Confusion Matrix")
conf_GBM = confusion_matrix(Y_test, predicted_GBC)
print(conf_GBM)

output_type = ['Spruce/Fir', 'Lodgepole', 'Ponderosa', 'Cottonwood', 'Aspen', 'Douglas', 'Krummholz']
df_cm_GBM = pd.DataFrame(conf_GBM, index=[i for i in output_type], columns=[i for i in output_type])
plt.figure(figsize=(10, 7))
sns.heatmap(df_cm_GBM, annot=True)
plt.show()

start_time = time.time()
classifierGBM = OneVsRestClassifier(
    GradientBoostingClassifier(n_estimators=200, max_features="auto", min_samples_leaf=1, random_state=7))
y_score_GBM = classifierGBM.fit(X_MM, Y_train).predict(X_test_MM)
elapsed_time = time.time() - start_time
print("Time taken: ", elapsed_time)

y_score_GBM_bin = label_binarize(y_score_GBM, classes=[1, 2, 3, 4, 5, 6, 7])

plot_ROC_curve('ROC curve for Gradient Boosting Machine Classifier',y_test_bin,y_score_GBM_bin)

# Time taken to run the complete program
elapsed_time = time.time() - start_time
print("Time taken to run program: ", elapsed_time)
