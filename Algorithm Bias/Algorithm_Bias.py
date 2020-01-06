import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import svm
from scipy import interp
from sklearn import tree
from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

# Load the dataset
bcDB = datasets.load_breast_cancer()
# create a df with features
bcDF = pd.DataFrame(bcDB.data, columns= list(bcDB['feature_names']))
# add a target column
bcDF['target'] = pd.Series(bcDB.target)
# sort by target
bcDF = bcDF.sort_values(by = ['target'])
# reset their index
bcDF = bcDF.reset_index(drop=True)
# show the top 5 entries
bcDF.head(5)

# count total for each label
vc = bcDF['target'].value_counts()
for i,j in enumerate(bcDB.target_names):
    print (vc[i],j)
    
# get label variable
y = bcDF.pop('target').values
# get feature variable
X = bcDF.values
# print dimensions
X.shape, y.shape




# CREATING CLASSIFIER FUNCTION

predicted = []
Names = []
# Takes a classifier name, set of features, set of labels,the classifier name, and 
def HoldOut(Classifiers,X, y, name):
        i = 0
        for classifier in Classifiers:
            y_count = y.mean()
            print('Y Set - 1: %.2f%%' % ((y_count)*100))
            print('Y Set - 0: %.2f%%' % ((1-y_count)*100))
            # Setup 70% training sets and 30% test sets from the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
            y_train_count = y_test.mean()
            print('Y Training Set - 1: %.2f%%' % ((y_train_count)*100))
            print('Y Training Set - 0: %.2f%%' % ((1-y_train_count)*100))
            # predict the test set
            y_pred = classifier.fit(X_train, y_train).predict(X_test)
            # Create an ROC graph and AUC calculation
            roc_auc = roc_auc_score(y_test, y_pred)
            Names.append(name[i])
            fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
            # Plot the ROC
            plt.figure()
            # lbael for AUC
            plt.plot(fpr, tpr, label='(%s %.2f%%)' % ( (name[i]), ((roc_auc)*100)))
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC %s Holdout' % name)
            plt.legend(loc='right')
            plt.show()
            print('Classification Parameters: %s' % classifier)
            #Confusion matrix for the classifier
            matrix = confusion_matrix(y_test, y_pred)
            print(matrix)
            TP = matrix[1, 1] # correctly diagnosed BC
            TN = matrix[0, 0] # correctly passed BC check
            FP = matrix[0, 1] # incorrectly passed BC check
            FN = matrix[1, 0] # incorrectly diagnosed BC
            classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
            classification_error = (FP + FN) / float(TP + TN + FP + FN)
            sensitivity = TP / float(FN + TP)
            specificity = TN / (TN + FP)
            false_pos_rate = FP / float(TN + FP)
            precision = TP / float(TP + FP)
            F1 = (((2*precision*sensitivity)/(precision+sensitivity))*100)
            print('True 0:%.2f%%' % ((1-y_test.mean())*100))
            print('Predicted 0:%.2f%%' % ((1-y_pred.mean())*100))
            print('True 1:%.2f%%' % ((y_test.mean())*100))
            print('Predicted 1:%.2f%%' % ((y_pred.mean())*100))
            print('Change: %.2f%%' % (((y_test.mean())*100) - ((y_pred.mean())*100)))
            predicted.append((((y_test.mean())*100) - ((y_pred.mean())*100)))
            print('classification_accuracy: %.2f%%' % ((classification_accuracy)*100))
            print('classification_error: %.2f%%' % ((classification_error)*100))
            print('sensitivity: %.2f%%' % ((sensitivity)*100))
            print('specificity: %.2f%%' % ((specificity)*100))
            print('false_pos_rate: %.2f%%' % ((false_pos_rate)*100))
            print('precision: %.2f%%' % ((precision)*100))
            print('F1: %.2f%%' % F1)
            i += 1


# CREATING CROSS VALIDATION FUNCTION
# K Fold function
# Takes a classifier name, set of features, and set of labels
def KFolder(Classifier,X, y):
    y_count = y.mean()
    print('Y Set - Benign: %.2f%%' % ((y_count)*100))
    print('Y Set - Malignant: %.2f%%' % ((1-y_count)*100))

    # Setup 70% training sets and 30% test sets from the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    y_train_count = y_test.mean()
    print('Y Training Set - Benign: %.2f%%' % ((y_train_count)*100))
    print('Y Training Set - Malignant: %.2f%%' % ((1-y_train_count)*100))

    # Run classifier with cross-validation and plot ROC curves
    cv = KFold(n_splits=5)
    
    # Most of the code below is from the sample on sklearn for Cross-Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = Classifier.fit(X_train, y_train).predict_proba(X_test)
        # ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Fold %d (%0.2f)' % (i, ((roc_auc)*100)))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean (%.2f%% $\pm$ %.2f%%)' % (((mean_auc)*100), std_auc),lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC KFold')
    plt.legend(loc='right')
    plt.show()


# CLASSIFIERS: HOLD-OUT STRATEGY
# Name for the graph
name = ['K-NN', 'Decision Tree', 'Logistic Regression', 'Naive Bayes']

# The classifier type
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
lr = LogisticRegression()
mnb = GaussianNB()

# Classifier list
classifiers = [knn, dtc, lr, mnb]
# Call the function passing the classifier type, features  (X), label (y), classifier name
HoldOut(classifiers, X, y, name)

# Change from imbalanced dataset to undersampling; both using defaulted classifiers 
for i in range(len(predicted)):
    print('%s: %.2f%%' % ((name[i]), (predicted[i])))
#   Building a k-Nearest Neighbours classifier with Holdout strategy. The main metric to look at here is how much of the test set was Malignant (minority classs, 32.75%), and how the classifier performed.
#Training Set of <font color=red>32.75%</font> is the amount of malignant.
#   This classifier predicted only <font color=red>30.99%</font> as Malignant, which means it has bias towards the Majority class.
#   The sensitivity shows <font color=red>89.57%</font> chance to predict a correct label of: Malignant.
#   The specificity shows <font color=red>85.71%</font> chance to predict a correct label of: Benign.

bcDB = datasets.load_breast_cancer()
bcDF = pd.DataFrame(bcDB.data, columns= list(bcDB['feature_names']))
bcDF['target'] = pd.Series(bcDB.target)
bcDF = bcDF.sort_values(by = ['target'])
bcDF = bcDF.reset_index(drop=True)

no_malignant = len(bcDF[bcDF['target'] == 1])
malignant = bcDF[bcDF.target == 0].index
random_indices = np.random.choice(malignant, no_malignant)
malignant_indices = bcDF[bcDF.target == 1].index
under_sample_indices = np.concatenate([malignant_indices, random_indices])
under_sample = bcDF.loc[under_sample_indices]

y = under_sample.pop('target').values
X = under_sample.values

# Name for the graph
name = ['K-NN Undersampling', 'Decision Tree Undersampling', 'Logistic Regression Undersampling', 'Naive Bayes Undersampling']

# The classifier type
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
lr = LogisticRegression()
mnb = GaussianNB()

# Classifier list
classifiers = [knn, dtc, lr, mnb]
# Call the function passing the classifier type, features (X), label (y), classifier name
HoldOut(classifiers, X, y, name)

rounded_true = []
rounded_pred = []
for i in range(int(len(predicted)/2)):
    rounded_true.append(round(predicted[i],2))
    rounded_pred.append(round(predicted[int((len(predicted)/2))+i],2))
    print('%s: %.2f%% -> %.2f%%' % ((Names[i]), (predicted[i]), (predicted[int((len(predicted)/2))+i])))

x = np.arange(len(Names)/2)
width = 0.25

fig, ax = plt.subplots(figsize=(10, 10))
rects1 = ax.bar(x - width/2, rounded_true, width, label='Default')
rects2 = ax.bar(x + width/2, rounded_pred, width, label='Undersampling')
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Majority Bias -> Minority Bias')
ax.set_title('Correct by Classifier')
ax.set_xticks(x)
ax.set_xticklabels(Names)
ax.legend()
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()


# OVERSAMPLING - ALL CLASSIFIERS
from collections import Counter
from sklearn.datasets import make_classification
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
for each in range(int(len(predicted)/2)):
    predicted = predicted[:-1]
    Names = Names[:-1]

bcDB = datasets.load_breast_cancer()
bcDF = pd.DataFrame(bcDB.data, columns= list(bcDB['feature_names']))
bcDF['target'] = pd.Series(bcDB.target)
bcDF = bcDF.sort_values(by = ['target'])
bcDF = bcDF.reset_index(drop=True)
bcDF.target.value_counts()

y = bcDF.pop('target').values
X = bcDF.values
print('Original dataset shape %s' % Counter(y))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

# Name for the graph
name = ['K-NN Oversampling', 'Decision Tree Oversampling', 'Logistic Regression Oversampling', 'Naive Bayes Oversampling']

# The classifier type
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
lr = LogisticRegression()
mnb = GaussianNB()

# Classifier list
classifiers = [knn, dtc, lr, mnb]
# Call the function passing the classifier type, features (X), label (y), classifier name
HoldOut(classifiers, X_res, y_res, name)

# GRAPH DEFAULT VS OVERSAMPLING
rounded_true = []
rounded_pred = []
for i in range(int(len(predicted)/2)):
    rounded_true.append(round(predicted[i],2))
    rounded_pred.append(round(predicted[int((len(predicted)/2))+i],2))
    print('%s: %.2f%% -> %.2f%%' % ((Names[i]), (predicted[i]), (predicted[int((len(predicted)/2))+i])))

x = np.arange(len(Names)/2)
width = 0.25
fig, ax = plt.subplots(figsize=(10, 10))
rects1 = ax.bar(x - width/2, rounded_true, width, label='Default')
rects2 = ax.bar(x + width/2, rounded_pred, width, label='Oversampling')
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Majority Bias -> Minority Bias')
ax.set_title('Correct by Classifier')
ax.set_xticks(x)
ax.set_xticklabels(Names)
ax.legend()
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.show()

"""
    CONCLUSION

While creating a model, it is always worth testing multiple accuracies:
A confusion matrix helps identify how the model predicted the true positive (predicted 1, and being a 1: TP), true negatives (predicted 0 and being 0: TN), false positives (predicted 1 but was actually 0: FP), and false negatives (predicted 0 but was 1: FN). These then allow you to calculate:
Classification accuracy = (TP + TN) / (TP + TN + FP + FN) - Accurately score the predicted data over the tested data
Classification error = (FP + FN) / (TP + TN + FP + FN) - How many incorrect predictions were made
Sensitivity = TP / (FN + TP) - How often did it correctly predict a 1
Specificity = TN / (TN + FP) - How often did it correctly predict a 0
False positive rate = FP / (TN + FP) - also known as FPR
Precision = TP / (TP + FP) - Also known as True Positive Rate, TPR
An ROC graph (TPR vs FPR) will enable you to calculate the Area Under the Curve (AUC), this visual quickly identifies a good model. The red line splitting the graph into 2 triangles is what a dummy model would predict, no model should be used if it is on or below this line. An ideal model will be as close to the upper left as possible.
I have also calculated the 'change' from the true set having a 0 (malignant) to the predicted set having a 0. This is what helped me the most in identifying bias as a binary classifier with 40% class A, and 60% class B is considered an imbalanced dataset. If a classifier is more bias towards B, the majority class, then we would expect something like 30% A and 70% B, showing a rise from A towards B; biast. 
There are multiple ways to correct this bias through dataset resampling; Oversampling and Undersampling. From my reading, oversampling is recommended on small dataset (as it increases the minority class to meet the majority class), and underrsampling is recommended on larger datasets (as it takes all the minority class but a slice of the majority class). In both cases the samples are then 50% / 50%. Therefore, any misclassifications that do happen are less likely to be from Bias, but probably from variance, in the Bias-Variance tradeoff.
"""