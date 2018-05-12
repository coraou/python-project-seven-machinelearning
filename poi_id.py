
# coding: utf-8

# In[61]:
# %load Untitled2222221.py

# In[3]:
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# data wrangling
import pandas as pd
import numpy as np

# feature selection
from sklearn.feature_selection import SelectKBest

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# validation
from sklearn.model_selection import KFold

# algorithms
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# tuning parameters
from sklearn.model_selection import GridSearchCV

# evaluation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

with open("../final_project_dataset.pkl/", "r") as data_file:  
    data_dict = pickle.load(data_file) 

my_dataset = data_dict

### Task 2: Remove outliers

#identify outlier
#import matplotlib.pyplot as plt
#features = ["salary", "bonus"]
#data = featureFormat(my_dataset, features)

#for point in data:
#    salary = point[0]
#    bonus = point[1]
#    plt.scatter( salary, bonus )

#plt.xlabel("salary")
#plt.ylabel("bonus")
#plt.show()

#remove outliers
my_dataset.pop("TOTAL", 0 )
my_dataset.pop("THE TRAVEL AGENCY IN THE PARK", 0 )
my_dataset.pop("LOCKHART EUGENE E", 0 )

#data-cleaning: filling in missing values with 0
enrons = pd.DataFrame.from_dict(my_dataset, orient = 'index')
enrons = enrons.apply(pd.to_numeric, errors='coerce')

def fillinginna(df_column):    
    df_column.fillna(value=0,inplace = True)
    #Convert them to int
    df_column = df_column.astype(int)
    
features_list = ['salary', 'total_payments', 'bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'restricted_stock', 
                  'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
#I removed features with more than 70 missing values from the dataframe

for col in features_list:
    fillinginna(enrons[col])

for col in features_list:
    enrons[col] = enrons[col].astype(int)

my_dataset = enrons.to_dict(orient='index')

### Task 3: Create a new feature
for en_row in my_dataset:
    from_this_person_to_poi = my_dataset[en_row]['from_this_person_to_poi']
    from_poi_to_this_person = my_dataset[en_row]['from_poi_to_this_person']
    from_messages = my_dataset[en_row]['from_messages']
    to_messages = my_dataset[en_row]['to_messages']
    if (from_messages != 0) and (to_messages != 0):
        my_dataset[en_row]['with_poi_proportion'] = float(from_this_person_to_poi + from_poi_to_this_person)/(from_messages + to_messages)
    else:
        my_dataset[en_row]['with_poi_proportion'] = 0
        
enrons = pd.DataFrame.from_dict(my_dataset, orient = 'index')

# Add a new feature to the feature list 
features_list = ['poi','salary', 'total_payments', 'bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'restricted_stock', 
                  'with_poi_proportion','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Feature selection
selector = SelectKBest(k = 12)
selector.fit(features, labels)
p_converted_scores = -np.log10(selector.pvalues_)
scores = selector.scores_
pd_scores = pd.DataFrame({'Features': features_list[1:],
                          'p-scores': p_converted_scores,
                          'features-scores': scores}).sort_values(by = 'p-scores',ascending=False)

#print pd_scores

# My final feature list
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'exercised_stock_options', 'shared_receipt_with_poi','expenses',
                 'restricted_stock', 'total_stock_value']
                 
# validation
labels = np.asarray(labels)
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(features):
    features_train, features_test = np.array(features)[train_index], np.array(features)[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
print features_train
 
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
#
##algorithm 1 decision tree

pipeline1 = [('scaler', MinMaxScaler()),('clf1', tree.DecisionTreeClassifier(random_state=42))]
pipe1 = Pipeline(pipeline1)
param_tree = dict(clf1__criterion=['gini', 'entropy'],
                  clf1__splitter=['random','best'],
                  clf1__max_depth=[1,10],
                  clf1__min_samples_leaf=[1,10],
                  clf1__min_impurity_decrease=[0,10],
                  clf1__min_samples_split=[2,40]
                  )
#from sklearn.model_selection import StratifiedShuffleSplit
#cv = StratifiedShuffleSplit(n_splits=1000)
grid_search1 = GridSearchCV(pipe1, scoring = 'recall', param_grid=param_tree)
grid_search1.fit(features_train, labels_train)
clf_tree = grid_search1.best_estimator_
pred1 = clf_tree.predict(features_test)

recall_score_tree = recall_score(labels_test,pred1,average='binary', pos_label=1)
precision_score_tree = precision_score(labels_test, pred1,average='binary', pos_label=1)
f1_score_tree = f1_score(labels_test, pred1)
tree = [recall_score_tree, precision_score_tree, f1_score_tree]

#
##algorithm 2 svms

pipeline2 = [('scaler', MinMaxScaler()), ('clf2', SVC(kernel = "rbf"))]
pipe2 = Pipeline(pipeline2)
param_svc = dict(clf2__C=[1.0,2.0])
grid_search2 = GridSearchCV(pipe2, param_grid=param_svc)
grid_search2.fit(features_train, labels_train)
clf_svm = grid_search2.best_estimator_
pred2 = clf_svm.predict(features_test)
#
recall_score_svms = recall_score(labels_test,pred2,average='binary', pos_label=1)
precision_score_svms = precision_score(labels_test, pred2,average='binary', pos_label=1)
f1_score_svms = f1_score(labels_test, pred2)
svms = [recall_score_svms, precision_score_svms, f1_score_svms]

#
##algorithm 3 naive bayes

pipeline3 = [('scaler', MinMaxScaler()), ('clf3', GaussianNB())]
pipe3 = Pipeline(pipeline3)
pipe3.fit(features_train,labels_train)
pred3 = pipe3.predict(features_test)
#
recall_score_nb = recall_score(labels_test,pred3,average='binary', pos_label=1)
precision_score_nb = precision_score(labels_test, pred3,average='binary', pos_label=1)
f1_score_nb = f1_score(labels_test, pred3)
nb = [recall_score_nb, precision_score_nb, f1_score_nb]


# Print out the evaluation scores of different algorithms to select the best algorithm
evaluation_pd = pd.DataFrame({'naive-bayes': nb,
                              'SVMS' : svms,
                              'decision-tree': tree},index = ['recall-score','precision-score','f1-score'])
print evaluation_pd 

#### Task 5: Tune your classifier to achieve better than .3 precision and recall 
#### using our testing script. Check the tester.py script in the final project
#### folder for details on the evaluation method, especially the test_classifier
#### function. Because of the small size of the dataset, the script uses
#### stratified shuffle split cross validation. For more info: 
#### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#
## Example starting point. Try investigating other evaluation techniques!
#
#
print ('The precision score of my algorithm is: {} , the recall score is: {} , the f1 score is:{}' ).format(precision_score_nb,recall_score_nb,f1_score_nb)
#### Task 6: Dump your classifier, dataset, and features_list so anyone can
#### check your results. You do not need to change anything below, but make sure
#### that the version of poi_id.py that you submit can be run on its own and
#### generates the necessary .pkl files for validating your results.
#
dump_classifier_and_data(clf_tree, my_dataset, features_list)





