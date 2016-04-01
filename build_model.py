import os

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from user_agents import parse

from featurizer import ClickstreamFeaturizer
from visualization import plotter

def create_file(input_path):
    my_directory = os.path.dirname(__file__)
    infile_raw = os.path.join(my_directory, input_path)
    infile = os.path.realpath(infile_raw)
    return infile

CLASS_LABELS = create_file('data/labeled_users_XXXXXXXX.tsv')
current_directory = create_file('data/')
INFILE = create_file('data/user_data_XXXXXXXX.csv')
OUTFILE = create_file('data/feature_matrix_XXXXXXXX.csv')
PALETTE = 'visualization/color_palette.json'
PR_CURVE = create_file('data/precision_recall_curve.png')

with open(PALETTE, 'r') as cp:
    colors = json.load(cp)

plotter = plotter.PythonPlotter(colors['colors_RGB'], alpha=0.7, fontsize=8, title_fontsize=14, \
                                                    directory=current_directory)

user_data = pd.read_csv(INFILE,header=0,parse_dates=['creation_time_search', \
                            'creation_time_session', 'creation_time_pageview'])

labels = pd.read_csv(CLASS_LABELS, header=0, delimiter='\t')
total_users = labels.shape[0]

PAGE_KEY_FILTERS = ('ssadminx', 'https:')
REF_UTM_TYPES = ('utm_source_cat', 'utm_medium_cat', 'utm_campaign_cat', \
                                            'referrer_cat', 'referrer_cleaned')
# remove non-informative page_keys
for filter_value in PAGE_KEY_FILTERS:
    user_data = user_data[user_data['page_key'].str.
        contains(filter_value)==False].reset_index(drop=True)

# Create lists of potential user attributes
ref_utm_params = defaultdict(list)
for ref_utm_type in REF_UTM_TYPES:
    ref_utm_params[ref_utm_type] = [x for x in user_data[ref_utm_type].unique() \
                                                    if pd.notnull(x) and x != '0']

page_keys = user_data['page_category'].dropna().unique()

# Populate featurizer with parameters
featurizer = ClickstreamFeaturizer(ref_utm_params, page_keys)

# Create feature name vector; add user_id as a join key
feature_name_vector = featurizer.get_feature_names()
feature_name_vector.append('user_id')

# Transform disaggreagate clickstream behavior into a usable feature matrix
user_feature_list = []
counter = -1
for uid, group in user_data.groupby('user_id'):
    counter += 1
    user_feature_list.append(featurizer.featurize(group)+[uid])
    if counter % 10000 == 0:
        print "Featurization in progress \n processed user {0}: ".format(counter)


nfeatures = set(list(matrix3.columns.values))
to_remove = filter(lambda x: x.startswith('Unnamed'), nfeatures )
to_remove += set(filter(lambda x: x.endswith('_id'), nfeatures))
nfeatures = list(nfeatures - to_remove)

feature_matrix = pd.DataFrame(user_feature_list, columns = feature_name_vector)
# save progress
feature_matrix.to_csv(OUTFILE, delimiter=',', header=0)

# start modeling with my two favorite classifers
X = matrix3[nfeatures]
y = matrix3['label']

mss = int(total_users * 0.005)
logistic_regression = linear_model.LogisticRegression()
random_forest = random_forest = RandomForestClassifier(n_estimators=100, \
    max_features="sqrt", n_jobs=-1, min_samples_split=mss, min_samples_leaf=5)

logistic_regression = linear_model.LogisticRegression()
random_forest = RandomForestClassifier()

# use Kfolds to understand confidence interval on predictions and prevent
# overfitting
# Use coeficients and feature importance to understand which features are most
# important in predicting user segments and future behavior

k_fold = cross_validation.KFold(len(X), 10, shuffle = True)
for k,(train,test) in enumerate(k_fold):
    random_forest.fit(X.ix[train],y.ix[train])
    print("[fold {0}], score: {1}".\
                    format(k,random_forest.score(X.ix[train],y.ix[train])))
    print "Features sorted by their score:"
    print sorted(zip(map(lambda x: round(x, 4), random_forest.feature_importances_), \
                                                    nfeatures), reverse=True)[:40]

k_fold = cross_validation.KFold(len(X), 10, shuffle = True)
for k,(train,test) in enumerate(k_fold):
    logistic_regression.fit(X.ix[train],y.ix[train])
    print("[fold {0}], score: {1}".\
                    format(k,logistic_regression.score(X.ix[train],y.ix[train])))

print 'Non-zero coefficients\n'
print 'Value\t\tFeature name'
coeffs = logistic_regression.coef_.ravel()
order = [i for i, coeff in sorted(enumerate(coeffs), key=lambda x:abs(x[1]), reverse=True)]
for i in order:
    print '%.3f\t\t%s' % (coeffs[i], nfeatures[i])
    if abs(coeffs[i]) < 0.001:
        break

# Compare classifer performance
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=.2)

score_LR = logistic_regression.fit(X_train, Y_train).decision_function(X_test)
score_RF = random_forest.fit(X_train, Y_train).predict_proba(X_test)[:,1]

precision_LR, recall_LR, _ = precision_recall_curve(Y_test, score_LR)
precision_RF, recall_RF, _ = precision_recall_curve(Y_test, score_RF)

x_list = [recall_RF, recall_LR]
y_list = [precision_RF, precision_LR]
title = 'Logistic Regression vs Random Forest'
label_list = ['Random Forest', 'Logistic Regression']
plotter.multiline(x_list, y_list, 'Recall', 'Precision', title, PR_CURVE, label_list)
