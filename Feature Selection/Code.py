"""
FEATURE SELECTION
The objective of this assignment is to assess the impact of feature selection on training and
test datasets.
"""

# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib notebook
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Create a df for training data and test data
train = pd.read_csv('SPECTF_train.csv')
test = pd.read_csv('SPECTF_test.csv')
print(train.shape)
print(test.shape)

# Setup the training and test X,Y's
X_train = train.drop("DIAGNOSIS", axis=1)
X_test = test.drop("DIAGNOSIS", axis=1)
y_train = train.DIAGNOSIS
y_test = test.DIAGNOSIS

# Use Gradient Boosting Classifier
model = GradientBoostingClassifier()

# Results for Training Data, using Cross-Validation 
kfold = KFold(n_splits=10, random_state=7)
Baseline_Training_accuracy = cross_val_score(model, X_train, y_train, cv=kfold)
print('Baseline Accuracy (Training, K-Fold): %.2f%%' % (Baseline_Training_accuracy.mean()*100))

# Results for Test Data, using Hold-Out
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
Baseline_Test_accuracy = accuracy_score(y_test, predictions)
print('Baseline Accuracy (Test, Hold-Out): %.2f%%' % (Baseline_Test_accuracy * 100.0))


# Calculate the Info Gain on each feature
mi = dict()
i_scores = mutual_info_classif(X_train, y_train)
for i,j in zip(train.columns,i_scores):
    mi[i]=j

# Store in a dataframe 
df = pd.DataFrame.from_dict(mi,orient='index',columns=['I-Gain'])
df.sort_values(by=['I-Gain'],ascending=False,inplace=True)

# Calculate the accuracy of each feature, incremental onto the model
acc_scores = []
for kk in range(1, X_train.shape[1]+1):
    Feature_Select_transform = SelectKBest(mutual_info_classif, k=kk).fit(X_train, y_train)
    X_tR_new = Feature_Select_transform.transform(X_train)
    X_tS_new = Feature_Select_transform.transform(X_test)
    spec_GBC = model.fit(X_tR_new, y_train)
    y_dash = spec_GBC.predict(X_tS_new)
    acc = accuracy_score(y_test, y_dash)
    acc_scores.append(acc)
df['Accuracy'] = acc_scores

# Plot the info gain and accuracy of adding each feature, we will then choose the best features
n = len(df.index)
rr = range(1,n)
fig, ax = plt.subplots(figsize=(10,5))
ax2 = ax.twinx()
ax.bar(df.index, df["I-Gain"], label='Info Gain',width=.35)
ax2.plot(df.index, df["Accuracy"], color='red', label='Accuracy')
ax.set_xticklabels(list(df.index), rotation = 90)
ax.set_xlabel('Features')
ax.set_ylabel('Information Gained')
ax2.set_ylabel('Accuracy')
ax.legend()
plt.show()

"""
    DISCUSSION
From the graph and table above, we can see that F20R provides the most information gain to the model, but the accuracy is not the best as we only have 1 feature. When we add the second highest info gain feature to the model, the accuracy improves. However, as we add more features, in order of their information gain (highest to lowest), we can see the accuracy of the model increase. Depending on processing power, you may choose to pick a lower set of features (7 appears good) as the accruacy tends to plateau around 10+ features. The model doesn't appear to go above a ~70% accuracy threshold, picking fewer features can help improve the accuracy and run time. We will look at the top 10 features by their info gain.
"""

features = df.head(10).index
X_train_scoring = pd.DataFrame()
X_test_scoring = pd.DataFrame()
y_train_scoring = train.DIAGNOSIS
y_test_scoring = test.DIAGNOSIS
for feature in features:
    X_train_scoring[feature] = X_train[feature].values
    X_test_scoring[feature] = X_test[feature].values

# Results for Training Data, using Cross-Validation 
kfold = KFold(n_splits=10, random_state=7)
IG_Training_accuracy = cross_val_score(model, X_train_scoring, y_train_scoring, cv=kfold)
print('Info Gain Accuracy (Training, K-Fold): %.2f%%' % (Baseline_Training_accuracy.mean()*100))

# Results for Test Data, using Hold-Out
model.fit(X_train_scoring, y_train_scoring)
y_pred = model.predict(X_test_scoring)
predictions = [round(value) for value in y_pred]
IG_Test_accuracy = accuracy_score(y_test_scoring, predictions)
print('Info Gain Accuracy (Test, Hold-Out): %.2f%%' % (Baseline_Test_accuracy * 100.0))

# WRAPPER-BASED FORWARD SEQUENTIAL SEARCH
#The Forward Seqeuntial Search will use Gradient Boost classifier and look at all the features added sequentially. Then, re-evaluate using the least amount of features which give the best accuracy.

# It doesn't appear to add any value past ~7 features, so change k_features to 7 if this runs slowly
sfs_forward = SFS(model,k_features=44,forward=True, verbose=1, scoring='accuracy',cv=10, n_jobs =-1)
sfs_forward = sfs_forward.fit(X_train, y_train)

# This will create a graphic that shows performance (accuracy) as a solid blue line for each feature added, 
# and the feint blue is the standard error for that feature
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs_forward.get_metric_dict(), kind='std_dev', figsize=(10, 5))
plt.ylim([0.5, 1])
plt.title('Sequential Forward Selection (Standard Error)')
plt.grid()
plt.show()

"""
    DISCUSSION
From the graph above, it would appear the 7 features would be the best model, after that the model performance again plateau's like with Information Gain. We will re-run the model using the 7 best features.
"""

# Rerun with 7 features
sfs_forward = SFS(model,k_features=7,forward=True, verbose=1, scoring='accuracy',cv=10, n_jobs =-1)
sfs_forward = sfs_forward.fit(X_train, y_train)

# Get the 7 features used
features = sfs_forward.k_feature_names_

# Create a new Dataframe with the 7 features
X_train_scoring2 = pd.DataFrame()
X_test_scoring2 = pd.DataFrame()
for feature in features:
    print(feature)
    X_train_scoring2[feature] = X_train[feature].values
    X_test_scoring2[feature] = X_test[feature].values
# Results for Training Data, using Cross-Validation  - Forward Search
kfold = KFold(n_splits=10, random_state=7)
FS_Training_accuracy = cross_val_score(model, X_train_scoring2, y_train_scoring, cv=kfold)
print('Forward Search Accuracy (Training, K-Fold): %.2f%%' % (FS_Training_accuracy.mean()*100))

# Results for Test Data, using Hold-Out
model.fit(X_train_scoring2, y_train_scoring)
y_pred = model.predict(X_test_scoring2)
predictions = [round(value) for value in y_pred]
FS_Test_accuracy = accuracy_score(y_test_scoring, predictions)
print('Forward Search Accuracy (Test, Hold-Out): %.2f%%' % (FS_Test_accuracy * 100.0))

#WRAPPER-BASED BACKWARD ELIMINATION
# Create a backward elimination model starting at 44 features (all of them) and ending at 1, this will help analyse which are the best 
# To avoid long performance time, can be set to 7 (or skipped), as I will be working with 7 (appeared to be the best on previous runs)
sbs_backward = SFS(model,k_features=1,forward=False, verbose=1, scoring='accuracy',cv=10, n_jobs =-1)
sbs_backward = sbs_backward.fit(X_train, y_train)

fig1 = plot_sfs(sbs_backward.get_metric_dict(),kind='std_dev',figsize=(6, 4))
plt.ylim([0.5, 1])
plt.title('Sequential Backward Elimination (Standard Error)')
plt.grid()
plt.show()

"""
The graph shows that removing the first ~7 (i.e. starting at 44 features and removing 7) improves the accuracy, it then plateaus with minor increases until the last ~7 features (i.e. working with 7 -> 1 features). For performance sake, we could remove the 7 features fom 44 and work on the remaining, or as a once off high performance, removed features until only 7 remain.
"""
# Again, 7 features appears to be best, but also removing 7 features works well (i.e. working with 37 features)
# Since Forward search worked on 7, I will look at 37 features here
sbs_backward_2 = SFS(model,k_features=37,forward=False, verbose=1, scoring='accuracy',cv=10, n_jobs =-1)
sbs_backward_2 = sbs_backward_2.fit(X_train, y_train)

fig2 = plot_sfs(sbs_backward_2.get_metric_dict(),kind='std_err', figsize=(6, 4))
plt.ylim([0.7, 1])
plt.title('Sequential Backward Elimination (Standard Error)')
plt.grid()
plt.show()

features2 = sbs_backward_2.k_feature_names_
features2
X_train_scoring3 = pd.DataFrame()
X_test_scoring3 = pd.DataFrame()
for feature in features2:
    X_train_scoring3[feature] = X_train[feature].values
    X_test_scoring3[feature] = X_test[feature].values

# Results for Training Data, using Cross-Validation - Backward Elimination
kfold = KFold(n_splits=10, random_state=7)
BE_Training_accuracy = cross_val_score(model, X_train_scoring3, y_train_scoring, cv=kfold)
print('Backward Elimination Accuracy (Training, K-Fold): %.2f%%' % (BE_Training_accuracy.mean()*100))

# Results for Test Data, using Hold-Out
model.fit(X_train_scoring3, y_train_scoring)
y_pred = model.predict(X_test_scoring3)
predictions = [round(value) for value in y_pred]
BE_Test_accuracy = accuracy_score(y_test_scoring, predictions)
print('Forward Search Accuracy (Test, Hold-Out): %.2f%%' % (BE_Test_accuracy * 100.0))

"""
    DISCUSSION
From the Backwards elimination graph, we can see that from 37 or higher features we lose performance, this then plateau's on performance until it reaches less than 7 features. We can interpret that the ideal number of features should lie between 7 and 37.
"""

# Round numbers to give better appearance in the graph
Baseline_test_round = round((Baseline_Test_accuracy * 100), 2)
IG_test_round = round((IG_Test_accuracy * 100), 2)
FS_test_round = round((FS_Test_accuracy * 100), 2)
BE_test_round = round((BE_Test_accuracy * 100), 2)
Baseline_training_round = round(((Baseline_Training_accuracy.mean()*100)), 2)
IG_training_round = round(((IG_Training_accuracy.mean()*100)), 2)
FS_training_round = round(((FS_Training_accuracy.mean()*100)), 2)
BE_training_round = round(((BE_Training_accuracy.mean()*100)), 2)

# Taken from matplotlib example
labels = ['Baseline', 'Info Gain', 'Forward', 'Backward']
Train = [Baseline_training_round, IG_training_round, FS_training_round, BE_training_round]
Test = [Baseline_test_round, IG_test_round, FS_test_round, BE_test_round]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10,5))
rects1 = ax.bar(x - width/2, Train, width, label='Train')
rects2 = ax.bar(x + width/2, Test, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Feature Selection')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
plt.show()

"""
    FEATURE SELECTION DISCUSSION
At a glance, the feature subset selection appears most accurate on backward selection. This could be because we have identified the last 7 features as least accurate and by removing them we have improved the overall accuracy of the model. This method also has more features than forward selection, but on the test data they do appear to perform similarily well, this could be because the accuracy plateau's after the first 7 features and we removed the (7) features that drops the accuracy. The main benefit between these two methods, for this data, would appear to be performance time. Information Gain appears to have dropped slightly compared to the baseline accuracy measure, probably due to the features being measured individually instead of as a sequential combination (as with Forward Selection/Backward Elimination).

    STABILITY & CONSISTENCY
For model stability, I looked at some evaluation metrics (https://github.com/nogueirs/JMLR2018/tree/master/python) that requires a feature matrix (or array) of size M (number of combinations) and length d (number of features) where 1 indicates the feature was selected and 0 indicates feature was not selected. This allows different feature combinations to be evaluated based on the variation of features selected as 'best features'. I couldn't get this to work so instead, I looked at the features being suggested by the model, ran it 10 times and counted the occurences of each feature. 

    CONCLUSION
From the feature subset selection methods I believe that forward selection provides the best measures in terms of both performance time (15 sec vs 25 sec for backward) and accuracy ( best of the 3 features).
"""

IG_df = pd.DataFrame()
features = []
for k in range (0,10):
    print('Running Iteration: %.0f' % (k+1))
    mi = dict()
    i_scores = mutual_info_classif(X_train, y_train)
    for i,j in zip(train.columns,i_scores):
        mi[i]=j

    # Store in a dataframe 
    df = pd.DataFrame.from_dict(mi,orient='index',columns=['I-Gain'])
    df.sort_values(by=['I-Gain'],ascending=False,inplace=True)
    features += df.head(10).index.tolist()
IG_df['IG'] = features
SFS_df = pd.DataFrame()
for k in range (0,10):
    print('Running Iteration: %.0f' % (k+1))
    sfs_forward = SFS(model, k_features=10, forward=True, verbose=1, scoring='accuracy',cv=10, n_jobs =-1)
    sfs_forward = sfs_forward.fit(X_train, y_train)
    features += sfs_forward.k_feature_names_
SFS_df['SFS'] = features

SBS_df = pd.DataFrame()
for k in range (0,10):
    print('Running Iteration: %.0f' % (k+1))
    sbs_backward = SFS(model,k_features=1,forward=False, verbose=1, scoring='accuracy',cv=10, n_jobs =-1)
    sbs_backward = sbs_backward.fit(X_train, y_train)
    features += sbs_backward.k_feature_names_
SBS_df['SBS'] = features

IG_features = IG_df['IG'].value_counts()
SFS_features = SFS_df['SFS'].value_counts()
SBS_features = SBS_df['SBS'].value_counts()

print('Information Gain:')
print(IG_features.head(5))
print('Forward Search:')
print(SFS_features.head(5))
print('Backward Elimination:')
print(SBS_features.head(5))

"""
From the above results, we can see that the first 5 features occur in each selection method but vary in the number of times it was picked. This indicates that the feature selection is fairly stable as the same 5 features are consistently in the top 5 features, but their order varies by only +/- one selection. For this reason, I believe that these methods are equally stable at feature selection, although minor variance implies Backward Elimination, Forward Search, and Information Gain are the most stable, respectively.
### Stability
(Feature, number of features it was included in)
Information Gain:
F12S    10
F20R    10
F6R     10
F20S    10
F15R    10

Forward Search:
F20S    20
F6R     16
F21R    13
F12S    12
F15R    10

Backward Elimination:
F20S    30
F6R     16
F21R    13
F12S    12
F15R    10
"""
