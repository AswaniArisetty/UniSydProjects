#!/usr/bin/python
"""
Helper functions to hold parameter values
and pipeline configuration 
used in each classifer.
"""
from sklearn.preprocessing import MinMaxScaler 
from sklearn.feature_selection import SelectKBest,SelectFromModel,f_classif,RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,SGDClassifier
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn.neighbors import KNeighborsClassifier 



def get_pipeline_params(clf_id): 
     '''Returns parameters for the classifier
     Args: 
         clf: Classifier name (LR -for Logistic Regression , 
                                SVC - Support Vector , 
                                LSVC - Linear Support Vector)
     Returns: 
         A dictionary of parameters to pass into an sk-learn grid-search  
             pipeline. 
     ''' 
     if clf_id =='LR' :
             params = {'selection__k':[15,16,17,18,19,'all'],
                           'classifier__C': [1e-3],
                           'classifier__class_weight': ['balanced'], 
                           'classifier__tol': [1e-16],
                           'pca__n_components': [2], 
                           'pca__whiten': [True] 
                           } 
     elif clf_id =='DTC' :
             params = {'selection__k':[20],
                           'classifier__class_weight': ['balanced'],
                           'classifier__max_features': ['sqrt'], 
                           'classifier__criterion':['entropy'],
                           'classifier__max_depth': [2],
                           'pca__n_components': [2], 
                           'pca__whiten': [True] 
                           } 
     elif clf_id =='SVC':
             params = {'selection__k':[20],
                           'classifier__C': [0.1],
                           'classifier__kernel': ['rbf'],
                           'classifier__gamma': ['auto'],
                           'classifier__class_weight': ['balanced'], 
                           'classifier__tol': [1e-3],
                           'pca__n_components': [2], 
                           'pca__whiten': ['False'] 
                           } 
     elif clf_id =='LSVC':
             params = {'selection__k':[20],
                           'classifier__C': [1e-6],
                           'classifier__class_weight': ['balanced'], 
                           'classifier__tol': [1e-16],
                           'pca__n_components': [2], 
                           'pca__whiten': [True] 
                           } 
     elif clf_id =='RFC':
             params = {'selection__k':[20],
                           'classifier__n_estimators': [1000],                           
                           'classifier__class_weight':['balanced'],
                           'classifier__max_features': [0.8], 
                           'classifier__n_jobs': [-1], 
                           #'classifier__min_samples_split': [10],
                           'pca__n_components': [2], 
                           'pca__whiten': [True] 
                           } 
     elif clf_id =='ABC':
             base_est=SVC(probability=True, kernel='linear'), 
             params = {'selection__k':[20],
                           'classifier__n_estimators': [1000],
                           'classifier__base_estimator': [SGDClassifier(loss='log', penalty='l1',n_jobs=-1,class_weight='balanced',l1_ratio=0.1)],
                           'classifier__algorithm': ['SAMME'],
                           #'lda__solver': ['svd'],
                           'pca__n_components': [2],                          
                           'pca__whiten': [True] 
                           } 
     elif clf_id =='VC': 
             params={'classifier__lr__C':[1e-5],
                     'classifier__lr__class_weight':['balanced'],
                     'classifier__lr__tol':[1e-32],
                     'classifier__svc__C':[0.1],
                     'classifier__svc__class_weight':['balanced'],
                     'classifier__svc__tol':[1e-3],
                     'classifier__svc__kernel':['rbf'],
                     'classifier__svc__gamma':['auto'],
                     'classifier__svc__random_state':[42],
                     'classifier__svc__probability':[True],
                     #'classifier__lsvc__C':[1e-6],
                     #'classifier__lsvc__class_weight':['balanced'],
                     #'classifier__lsvc__tol':[1e-32],
                     #'classifier__lsvc__random_state':[42],
                     'classifier__abc__n_estimators':[1000],
                     'classifier__abc__base_estimator':[SGDClassifier(loss='log', penalty='l1',n_jobs=-1,class_weight='balanced',l1_ratio=0.1)],
                     'classifier__abc__algorithm': ['SAMME'],
                     'selection__k':[15,16,17,18,19,'all'],
                     #'selection__n_features_to_select':[20],
                     'pca__n_components': [2], 
                     'pca__whiten': [True], 
                     #'lda__solver': ['lsqr'], 
                     'classifier__voting':['hard']
                     }
             
     return params 

def get_pipeline(clf_id): 
     '''Returns pipeline for given classifier
     Args: 
         clf: Classifier name (LR -for Logistic Regression , 
                                SVC - Support Vector , 
                                LSVC - Linear Support Vector)
     Returns: 
         Pipeline of stages including scaler,selection,pca and classifier
          to be passed to sk-learn grid-search  
     '''
     lr = LogisticRegression(C=1e-5,class_weight='balanced',tol=1e-32)
     skb=SelectKBest(score_func=f_classif)
     if clf_id =='LR' :
        pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), 
                                        ('selection', SelectKBest(score_func=f_classif)), 
                                        #('selection', RFE(estimator=lr, n_features_to_select=20, step=1)),
                                        ('pca', PCA()), 
                                        #('sampler1',EditedNearestNeighbours()),
                                        #('sampler2',RepeatedEditedNearestNeighbours()),
                                        ('classifier', LogisticRegression())
                                        ]) 
     elif clf_id =='SVC':
        pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), 
                                        ('selection', SelectKBest(score_func=f_classif)), 
                                        ('pca', PCA()), 
                                        ('classifier', SVC()) 
                                        ]) 
     elif clf_id =='DTC':
        pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), 
                                        ('selection', SelectKBest(score_func=f_classif)), 
                                        ('pca', PCA()), 
                                        ('classifier', DecisionTreeClassifier()) 
                                        ]) 
     elif clf_id =='LSVC':
        pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), 
                                        ('selection', SelectKBest(score_func=f_classif)), 
                                        ('pca', PCA()), 
                                        ('classifier', LinearSVC()) 
                                        ]) 
     elif clf_id =='RFC':
        pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), 
                                        ('selection', SelectKBest(score_func=f_classif)), 
                                        ('pca', PCA()), 
                                        ('classifier', RandomForestClassifier()) 
                                        ]) 
     elif clf_id =='ABC':
        pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), 
                                        ('selection', SelectKBest(score_func=f_classif)), 
                                        ('pca', PCA()), 
                                        ('classifier', AdaBoostClassifier()) 
                                        ]) 
     elif clf_id=='VC': 
           clf1=LogisticRegression()
           clf2=SVC()
           clf3=LinearSVC()
           clf4=AdaBoostClassifier()
           clf5=GaussianNB()
           pipeline=Pipeline([('scaler', MinMaxScaler()), 
                                        ('selection', SelectKBest(score_func=f_classif)), 
                                        #('selection', RFE(estimator=LogisticRegression(), n_features_to_select=20, step=1)),
                                        ('pca', PCA()), 
                                        ('classifier', VotingClassifier(estimators=[('lr',clf1),('svc',clf2),('abc',clf4)],weights=[1,1,1])) ])
     return pipeline 
