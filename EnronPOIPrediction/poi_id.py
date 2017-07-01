#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("tools/")

from operator import itemgetter
from time import time
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support,precision_score,recall_score
from feature_format import featureFormatPandas, targetFeatureSplitPandas
from poi_pipelines_params import get_pipeline_params,get_pipeline
from tester import dump_classifier_and_data,test_classifier
from sklearn.ensemble import ExtraTreesClassifier


### Task 1: Select what features you'll use.
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### A pandas dataframe for easier processing
enron_df=pd.DataFrame.from_dict(data_dict, orient='index') 
del enron_df['email_address']

#enron_df=enron_df[['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']]
#enron_df=enron_df[['poi','salary','bonus','total_payments','exercised_stock_options','to_messages','from_messages','shared_receipt_with_poi', 'total_stock_value','from_this_person_to_poi','from_poi_to_this_person']]

### Task 2: Remove outliers
### Removing 'TOTAL' as it seems to be an entry by mistake
### all zero rows and replace NaN string with np.NaN
enron_df=enron_df.drop('TOTAL',axis=0)
enron_df=enron_df.drop('LOCKHART EUGENE E',axis=0)
enron_df=enron_df.replace('NaN',np.NaN)
enron_df=featureFormatPandas(enron_df,remove_all_zeroes=True,replace_NaN=True)

### Task 3: Create new feature(s)
### totals and ratios of few attributes 
#enron_df['total_money_value'] = enron_df['total_payments'] + \
#                                enron_df['total_stock_value'] 


enron_df['total_poi_interaction'] = enron_df['shared_receipt_with_poi'] + \
                                    enron_df['from_this_person_to_poi'] + \
                                    enron_df['from_poi_to_this_person'] 

enron_df['mails_to_poi_ratio']=enron_df['from_this_person_to_poi'].div(enron_df['from_messages'])
enron_df['mails_from_poi_ratio']=enron_df['from_poi_to_this_person'].div(enron_df['to_messages'])

enron_df=featureFormatPandas(enron_df,remove_all_zeroes=True,replace_NaN=True)

### Extract features and labels from dataset for local testing

labels,features=targetFeatureSplitPandas(enron_df)
features_list=['poi']+list(features.columns)

## __name__ must == '__main__' to run parallel processing (n_jobs=-1 in GridSearchCV) in Windows 
if __name__ == "__main__":
#       Cross validation with 10% test data and random_state = 42 
#       as used in the tester script
        sss_cv = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1,random_state=42)
###############################################################################################
#       refer get_pipeline and get_pipeline_params in poi_pipelines_params
#       VC - Voting Classifier , LR - LogisticRegression , SVC - SVC , LSVC - LinearSVC
#       DTC - DecisionTreeClassifier , ABC - AdaBoostClassifier 
###############################################################################################
        pipeline = get_pipeline('VC')
        params = get_pipeline_params('VC')
        scoring_metric = 'recall' 
        grid_searcher = GridSearchCV(pipeline, param_grid=params,cv=sss_cv,
                                    n_jobs=-1, scoring=scoring_metric, verbose=1) 
        
        grid_searcher.fit(features, labels) 
################################################################################################
#        Uncomment below lines to print the parameters used and features selected              
        selected_cols = grid_searcher.best_estimator_.named_steps['selection'].get_support() 
        top_features = [x for (x, boolean) in zip(features.columns, selected_cols) if boolean] 
#        n_pca_components = grid_searcher.best_estimator_.named_steps['pca'].n_components_ 
         
#        print "{0} score: {1}".format(scoring_metric, grid_searcher.best_score_) 
#        print "{0} features selected".format(len(top_features))
        print "best features selected :\n",  top_features      
#        print "{0} PCA components".format(n_pca_components) 
#
#        Print the parameters used in the model selected from grid search 
#        print "Params: ", grid_searcher.best_params_  
##################################################################################################
        
        #print feature_importances 
        clf = grid_searcher.best_estimator_ 
        features.insert(0,'poi',labels)
        my_dataset=features.transpose().to_dict()
        
        test_classifier(clf,my_dataset,features_list)
        
        
        
        dump_classifier_and_data(clf, my_dataset, features_list)