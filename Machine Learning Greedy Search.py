#!/usr/bin/env python
# coding: utf-8

# #Machine Learning Team Project by Max Needle, Vincent Lo, Ranee Patel, Lucius Pham, and Mengshi Wang

# ## SIOP Competition
# 
# Our team entered the annual machine learning competition run by the Society for Industrial Organizational Psychology (SIOP). Using Walmart candidate screening data, we sought to predict which half of a development (test) applicant pool would represent the highest amount of true High Performers (worth 25% of our final score), the highest amount of true Retained employees (worth 25% of our final score), and the highest amount of both true Retained High Performers (worth 50% of our final score). We also had to account for adverse impact, the difference in hiring rates between Protected applicants (belonging to any of a number of protected classes) and Non-Protected applicants, as a penalty for "unfairness" in our selection method would be subtracted from our final score.
# 
# In this notebook, we detail two strategies that we employed. First, we created a binary variable for whether a candidate was both Retained and a High Performer (1) or not (0). We only used the subset of training data in which both of these variables were known in addition to whether they belonged to a Protected group or not. 
# 
# Then we separated this training dataset into two groups: Protected and Non-Protected (as per Dr. Squires' suggestion). For each of the two groups, we imputed missing data in three different ways (mean, median, and KNN with 10 neighbors), giving us 3 imputed datasets for each of the two groups. Then we used five-fold cross-validation on multiple models and we found the best performing model for each of the 2 groups (across all three imputations) according to the AUC of the models. 
# 
# In the first strategy, we used the top performing models to predict the probability that a Protected applicant and a Non-Protected applicant (separately based on their individual best models) would be a "Retained High Performer". 
# 
# In order to determine which half of this training applicant pool should be hired, we used their predicted probabilities of being a "Retained High Performer" to find the combination of Protected and Non-Protected applications which would result in the highest final score. We then created a new Hire/Not Hire variable, on which this half of the applicant pool were deemed as hires and the rest were not. Finally, we trained the development (test) applicant pool based on this Hire/Not Hire variable and chose the top 50% of that pool with the highest probability of being hires. We then submitted our hiring determinations to the competition organizers, and we were told that this strategy earned a final score of 46.1747. 
# 
# In the second strategy, we decided to use the top performing models for Protected and Non-Protected training applicants to directly predict the probability that a development Protected applicant and a development Non-Protected applicant (separately based on their individual best models) would be a Retained High Performer. In order to determine which half of this development applicant pool should be hired, we used their predicted probabilities of being Retained High Performers to find the combination of Protected and Non-Protected applicants which would result in the highest final score. We then submitted our hiring determinations to the competition organizers, and we were told that this strategy earned a final score of 56.5861.
# 
# The higher of these two final scores, 56.5861, put us at 16th out of 24 teams on the leaderboard in the competition at the time of submission.

# ### Import packages and set up display options

# In[50]:


# import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from scipy.stats import zscore
from scipy import interp

from sklearn import svm, datasets, tree
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[51]:


# set max view
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_colwidth',1000)


# ## Import training dataset, isolate instances with all necessary DVs ("Retained", "High_Performer", and "Protected_Group"), and create binary variable of "Retained_High_Performer"

# In[53]:


#Link the dataset from github

url_train = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/datasets/train.xlsx'

# import data and drop columns with id and unhelpful criterion values

data = pd.read_excel((url_train), na_values= ' ')


# In[54]:


# isolate 7890 instances with data all necessary DVS
training = data[((data['Retained'].notnull()) & (data['High_Performer'].notnull()) & (data['Protected_Group'].notnull()))]


# In[55]:


# create "Retained_High_Performer"
training['Retained_High_Performer']= [1 if ((training.iloc[row,6]==1) & (training.iloc[row,8]==1)) else 0 for row in range(len(training))]


# ## Split training dataset based on "Protected_Group" (Protected and Non-Protected) and impute missing values with mean, median, and KNN with 10 neighbors (using knn.impute in the bnstruct package in R) separately to produce three different imputed datasets for each group (Protected and Non-Protected)
# 

# In[56]:


# split training data into Protected and Non-Protected
protected = training[training['Protected_Group']==1].reset_index()
non_protected = training[training['Protected_Group']==0].reset_index()


# In[57]:


# impute features of Protected with mean, median, and KNN
protected_X = protected.loc[:, 'SJ_Most_1':'PScale13_Q5']

mean_protected_X = protected_X.fillna(protected_X.mean())
median_protected_X = protected_X.fillna(protected_X.median())

# protected_X.to_excel('protected_X_to_impute.xlsx', index = False)

# this is the code I ran in R to impute KNN
# library(readxl)
# X_to_impute <- read_excel("protected_X_to_impute.xlsx")
# library(bnstruct)
# KNN_X = knn.impute(as.matrix(X_to_impute), k= 10)
# library(writexl)
# write_xlsx(as.data.frame(KNN_X), "~/Desktop/SIOP Analysis/protected_X_imputed.xlsx")

url_KNNp = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/datasets/protected_X_imputed.xlsx'

KNN_protected_X= pd.read_excel(url_KNNp)


# In[58]:


# impute features of Protected with mean, median, and KNN
non_protected_X = non_protected.loc[:, 'SJ_Most_1':'PScale13_Q5']

mean_non_protected_X = non_protected_X.fillna(non_protected_X.mean())
median_non_protected_X = non_protected_X.fillna(non_protected_X.median())

# non_protected_X.to_excel('non_protected_X_to_impute.xlsx', index = False)

# this is the code I ran in R to impute KNN
# library(readxl)
# X_to_impute <- read_excel("non_protected_X_to_impute.xlsx")
# library(bnstruct)
# KNN_X = knn.impute(as.matrix(X_to_impute), k= 10)
# library(writexl)
# write_xlsx(as.data.frame(KNN_X), "~/Desktop/SIOP Analysis/non_protected_X_imputed.xlsx")

url_KNNnp = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/datasets/non_protected_X_imputed.xlsx'

KNN_non_protected_X= pd.read_excel(url_KNNnp)


# ## PROTECTED: Using 5-fold cross-validation on multiple models (5 Logistic Regression, 1 Naive Bayes, 1 Decision Tree, 7 Random Forests, 1 Support Vector Machine, 48 Neural Networks, 3 KNN models, and 243 XGBoost), find the best performing model for the Protected group (across all three imputations) according to the AUC of the models.

# In[59]:


# this model takes an (imputed) X dataframe, a y variable, and the type of imputation (String)
def cross_validated_models(X, y, X_imputation):
    
    X = X.apply(zscore)

    # create a dataframe to hold the model results
    df = pd.DataFrame(columns= ['Model','Parameters', 'Accuracy', 'Recall','ROC_AUC'])
        
    # logistic regression
    model = "Logistic Regression"
    # for each of 5 solver parameters
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    for s in solvers:
        # run 5 logistic regression models with the normalized X dataframe and average the cross validation scores
        score = pd.DataFrame(cross_validate(LogisticRegression(solver = s, max_iter=1000, random_state=0), X, y, scoring = ['recall','accuracy','roc_auc']))[['test_accuracy','test_recall','test_roc_auc']].mean()
        # add model results to the overall dataframe
        df = df.append({'Model': model,'Parameters': 'solver: '+s, 'Accuracy': score[0],'Recall': score[1], 'ROC_AUC': score[2]}, ignore_index = True)  

    # naive bayes
    model = "Naive Bayes"
    score = pd.DataFrame(cross_validate(GaussianNB(), X, y, scoring = ['recall','accuracy','roc_auc']))[['test_accuracy','test_recall','test_roc_auc']].mean()
    df = df.append({'Model': model,'Parameters': 'none', 'Accuracy': score[0],'Recall': score[1], 'ROC_AUC': score[2]}, ignore_index = True)

    # decision tree
    model = "Decision Tree"    
    score = pd.DataFrame(cross_validate(tree.DecisionTreeClassifier(random_state=0), X, y, scoring = ['recall','accuracy','roc_auc']))[['test_accuracy','test_recall','test_roc_auc']].mean()
    df = df.append({'Model': model,'Parameters': 'none', 'Accuracy': score[0],'Recall': score[1], 'ROC_AUC': score[2]}, ignore_index = True)

    # random forest
    model = "Random Forest"
    estimators = [150,300,450,600,750,900,1050]
    for e in estimators:
        for m in max_features:
            score = pd.DataFrame(cross_validate(rfc(n_estimators=e, random_state=0), X, y, scoring = ['recall','accuracy','roc_auc']))[['test_accuracy','test_recall','test_roc_auc']].mean()
            df = df.append({'Model': model,'Parameters': 'n_estimators: '+str(e), 'Accuracy': score[0],'Recall': score[1], 'ROC_AUC': score[2]}, ignore_index = True)

    # support vector machine (linear SVM)
    model = "Support Vector Machine"
    score = pd.DataFrame(cross_validate(LinearSVR(random_state=0), X, y, scoring = 'roc_auc')).mean()['test_score']
    df = df.append({'Model': model,'Parameters': 'none', 'Accuracy': np.nan,'Recall': np.nan, 'ROC_AUC': score}, ignore_index = True)

    # neural networks
    model = "Neural Network"
    hidden_layer_sizes=[(120,),(120,120),(120,120,120),(120,120,120,120)]
    activation= ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    for h in hidden_layer_sizes:
        for a in activation:
            for s in solver:
                score = pd.DataFrame(cross_validate(MLPClassifier(hidden_layer_sizes = h, activation = a, solver = s, max_iter= 1000, random_state=0), X, y, scoring = ['recall','accuracy','roc_auc']))[['test_accuracy','test_recall','test_roc_auc']].mean()
                df = df.append({'Model': model,'Parameters': 'hidden_layer_sizes: '+str(h)+' & activation: '+a+' & solver: '+s, 'Accuracy': score[0],'Recall': score[1], 'ROC_AUC': score[2]}, ignore_index = True)

    # KNN
    model = "KNN"
    neighbors = [3,5,10]
    for n in neighbors:
        score = pd.DataFrame(cross_validate(KNeighborsClassifier(n_neighbors=n), X, y, scoring = ['recall','accuracy','roc_auc']))[['test_accuracy','test_recall','test_roc_auc']].mean()
        df = df.append({'Model': model,'Parameters': 'n_neighbors: '+str(n), 'Accuracy': score[0],'Recall': score[1], 'ROC_AUC': score[2]}, ignore_index = True)

    #XGBoost
    model ="XGBoost"
    learning_rate = [.01, .03, .05]
    n_estimators = [100, 300, 500]
    max_depth = [5, 10, 15]
    min_child_weight = [1, 3, 5]
    gamma = [.1, .2, .3]
    for l in learning_rate:
        for n in n_estimators:
            for m in max_depth:
                for c in min_child_weight:
                    for g in gamma:
                        score = pd.DataFrame(cross_validate(xgb.XGBClassifier(subsample=.8, colsample_bytree=.8, reg_alpha=1e-5, learning_rate=l, n_estimators=n, max_depth=m, child_weight=c, gamma=g, random_state=0), X, y, scoring = ['recall','accuracy','roc_auc']))[['test_accuracy','test_recall','test_roc_auc']].mean()
                        df = df.append({'Model': model,'n_estimators': (n),'learning_rate': (l),'max_depth': (m),'min_child_weight': (c),'gamma': (g),'Accuracy': score[0],'Recall': score[1], 'ROC_AUC': score[2]}, ignore_index = True)

    df['Imputation']= X_imputation
    print("It ran!")
    return df


# In[60]:


# run models to predict for protected with median-, mean-, and KNN-imputed data

# protected_y = protected.loc[:,"Retained_High_Performer"]

# mean_protected= cross_validated_models(mean_protected_X, protected_y, 'Mean')
# median_protected= cross_validated_models(median_protected_X, protected_y, 'Median')
# KNN_protected= cross_validated_models(KNN_protected_X, protected_y, 'KNN')

# find and save best model from results
# protected_results = pd.concat([mean_protected,median_protected,KNN_protected], ignore_index=True)
# protected_results.sort_values(by=['ROC_AUC'], ascending=False).head(1)
# protected_results.to_excel('protected_results.xlsx', index = False)

# best model found for protected: 
# mean-imputed Neural Network with hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd" (mean ROC AUC of 0.611)

url_protected_results = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/models/protected_results.xlsx'

protected_results= pd.read_excel(url_protected_results)


# ### AUC visualization the for best model for protected group

# In[62]:


# re-run the best model of : mean-imputed Neural Network with hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd" (mean ROC AUC of 0.611)

def best_modelp(X, y, X_imputation):
    
    X = X.apply(zscore)

    df_best = pd.DataFrame(columns= ['Model','Parameters', 'Accuracy', 'Recall','ROC_AUC'])

    score_best = pd.DataFrame(cross_validate(MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd", max_iter= 1000, random_state=0), X, y, scoring = ['recall','accuracy','roc_auc'], cv=5))[['test_accuracy','test_recall','test_roc_auc']].mean()

    df_best = df_best.append({'Model': "Neural Network",'Parameters': 'n_neighbors: '+str(120,), 'Accuracy': score_best[0],'Recall': score_best[1], 'ROC_AUC': score_best[2]}, ignore_index = True)
    
    return df_best

protected_y = protected.loc[:,"Retained_High_Performer"]

best_modelp(mean_protected_X,protected_y, 'Mean')


# In[63]:


# plot the AUC for the best model

protected_y = protected.loc[:,"Retained_High_Performer"]

X = mean_protected_X.values
Y = protected_y.values

cv = StratifiedKFold(n_splits=5)
classifier = MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd", max_iter= 1000,random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X, Y):
    probas_ = classifier.fit(X[train], Y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC of Neural Network',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()


# ## NON-PROTECTED: Using 5-fold cross-validation on multiple models (5 Logistic Regression, 1 Naive Bayes, 1 Decision Tree, 7 Random Forests, 1 Support Vector Machine, 48 Neural Networks, 3 KNN models, and 243 XGBoost), find the best performing model for the Protected group (across all three imputations) according to the AUC of the models.

# In[64]:


# run models to predict for non-protected with median-, mean-, and KNN-imputed data

# non_protected_y = non_protected.loc[:,"Retained_High_Performer"]

# mean_non_protected= cross_validated_models(mean_non_protected_X, non_protected_y, 'Mean')
# median_non_protected= cross_validated_models(median_non_protected_X, non_protected_y, 'Median')
# KNN_non_protected= cross_validated_models(KNN_non_protected_X, non_protected_y, 'KNN')

# find and save best model from results
# non_protected_results = pd.concat([mean_non_protected,median_non_protected,KNN_non_protected], ignore_index=True)
# non_protected_results.sort_values(by=['ROC_AUC'], ascending=False).head(1)
# non_protected_results.to_excel('non_protected_results.xlsx', index = False)

# best model for non_protected: 
# KNN-imputed Neural Network with hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd" (mean ROC AUC of 0.602)

url_non_protected_results = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/models/non_protected_results.xlsx'

non_protected_results= pd.read_excel(url_non_protected_results)


# ### AUC visualization for best model for non-protected group

# In[65]:


# run the best model of: KNN-imputed Neural Network with hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd" (mean ROC AUC of 0.602)

def best_modelnp(X, y, X_imputation):
    
    X = X.apply(zscore)

    df_bestnp = pd.DataFrame(columns= ['Model','Parameters', 'Accuracy', 'Recall','ROC_AUC'])

    scorenp = pd.DataFrame(cross_validate(MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd", max_iter= 1000), X, y, scoring = ['recall','accuracy','roc_auc'], cv=5))[['test_accuracy','test_recall','test_roc_auc']].mean()

    df_bestnp = df_bestnp.append({'Model': "Neural Network",'Parameters': 'n_neighbors: '+str(120,), 'Accuracy': scorenp[0],'Recall': scorenp[1], 'ROC_AUC': scorenp[2]}, ignore_index = True)
    
    return df_bestnp

non_protected_y = non_protected.loc[:,"Retained_High_Performer"]

best_modelnp(KNN_non_protected_X,non_protected_y, 'KNN')


# In[66]:


# plot the AUC for the best model

non_protected_y = non_protected.loc[:,"Retained_High_Performer"]

X2 = KNN_non_protected_X.values
Y2 = non_protected_y.values

cv = StratifiedKFold(n_splits=5)
classifier = MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd", max_iter= 1000)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X2, Y2):
    probas_ = classifier.fit(X2[train], Y2[train]).predict_proba(X2[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y2[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC of Neural Network',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()


# # Strategy 1: Create the Hire/Not Hire variable in the training dataset and then train the development dataset on that.
# 
# 
# 

# ## PROTECTED: Use the top performing model to predict the probabilities of being a "Retained High Performer" for Protected training applicants.

# In[67]:


# train the top performing model (with x normalized) and predict probabilities for training dataset (with x normalized)
protected_y = protected.loc[:,"Retained_High_Performer"]
protected_predicted_proba = MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd").fit(KNN_protected_X.apply(zscore), protected_y).predict_proba(KNN_protected_X.apply(zscore))


# In[68]:


protected['Probability_of_Retained_High_Performer'] = pd.DataFrame(protected_predicted_proba)[1]


# ## NON-PROTECTED: Use the top performing model to predict the probabilities of being a "Retained High Performer" for Non-Protected training applicants.

# In[69]:


# train the top performing model (with x normalized) and predict probabilities for training dataset (with x normalized)
non_protected_y = non_protected.loc[:,"Retained_High_Performer"]
non_protected_predicted_proba = MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd").fit(mean_non_protected_X.apply(zscore), non_protected_y).predict_proba(mean_non_protected_X.apply(zscore))


# In[70]:


non_protected['Probability_of_Retained_High_Performer'] = pd.DataFrame(non_protected_predicted_proba)[1]


# ## Organize the Protected training probabilities in ascending order and concatenate them onto the Non-Protected training probabilities organized in descending order

# In[71]:


columns = ['UNIQUE_ID','High_Performer','Protected_Group','Retained','Retained_High_Performer','Probability_of_Retained_High_Performer']

protected_ascending = protected.sort_values(by=['Probability_of_Retained_High_Performer'], ascending = True).loc[:,columns]
non_protected_descending = non_protected.sort_values(by=['Probability_of_Retained_High_Performer'], ascending = False).loc[:,columns]


# In[72]:


ordered_results = pd.concat([protected_ascending,non_protected_descending], ignore_index=True).reset_index()


# ## Use a for-loop to go through all adjacent combinations of half of the training dataset (a training dataset of 7890 total applicants requires that 3945 applicants be hired), calculating adverse impact, true retained employees, true high perfomers, and true retained high performers, and then using these values to calculate the final score (somewhat like gradient descent but calculated all at once)

# In[73]:


# create dataframe with column for unfairness
protected_total = len(protected)
protected_list = list(ordered_results['Protected_Group'])
unfairness_list = [] 
for i in range(len(ordered_results)-3945):
    protected_identified = sum(protected_list[i:i+3945])
    protected_hire_rate = protected_identified/protected_total
    non_protected_hire_rate = (3945-protected_identified)/(7890-protected_total)
    unfairness = abs(1-(protected_hire_rate/non_protected_hire_rate))
    unfairness_list += [unfairness]

output = pd.DataFrame({'Unfairness':unfairness_list})


# In[74]:


# create new column for percent high_performer identified 
high_performer_total = len(ordered_results[ordered_results['High_Performer']==1.0])
high_performer_list = list(ordered_results['High_Performer'])
high_performer_identified_list = [] 
for i in range(len(ordered_results)-3945):
    high_performer_identified = sum(high_performer_list[i:i+3945])/high_performer_total
    high_performer_identified_list += [high_performer_identified]

output['Percent_High_Performer']= high_performer_identified_list


# In[75]:


# create new column for percent retained identified 
retained_total = len(ordered_results[ordered_results['Retained']==1])
retained_list = list(ordered_results['Retained'])
retained_identified_list = [] 
for i in range(len(ordered_results)-3945):
    retained_identified = sum(retained_list[i:i+3945])/retained_total
    retained_identified_list += [retained_identified]

output['Percent_Retained']= retained_identified_list


# In[76]:


# create new column for percent retained_high_performer identified 
retained_high_performer_list = list(ordered_results['Retained_High_Performer'])
retained_high_performer_total = len(ordered_results[ordered_results['Retained_High_Performer']==1])
retained_high_performer_identified_list = [] 
for i in range(len(ordered_results)-3945):
    retained_high_performer_identified = sum(retained_high_performer_list[i:i+3945])/retained_high_performer_total
    retained_high_performer_identified_list += [retained_high_performer_identified]

output['Percent_Retained_High_Performer']=retained_high_performer_identified_list


# In[77]:


# create column for final_score
output['Final_Score'] = 0.5*output['Percent_Retained_High_Performer']+0.25*output['Percent_Retained']+0.25*output['Percent_High_Performer']-output['Unfairness']


# ## Find the combination that results in the maximum final score and create a new "Hire" variable denoting that this half of applicants should be hired and the other half should not

# In[78]:


# max final_score
max_index = output[output['Final_Score']==max(output['Final_Score'])].index
output.loc[max_index,:]


# In[79]:


#  create a column called 'Hire' and fill it with 1s for the 3945 included in the max final_score distribution
hire=[]
for i in range(len(ordered_results)):
    if ((i > max_index) & (i <= max_index+3945)):
        hire += [1]
    else:
        hire += [0]

ordered_results['Hire'] = hire


# ## Return to the original training dataset that only contained instances with values for all three DVs ("Retained", "High_Performer", and "Protected_Group") but was not split by Protected_Group yet. Impute missing values with mean, median, and KNN with 10 neighbors (using knn.impute in the bnstruct package in R) to produce three different imputed datasets

# In[80]:


# impute features of training with mean, median, and KNN
training_X = training.loc[:, 'SJ_Most_1':'PScale13_Q5']

mean_training_X = training_X.fillna(training_X.mean())
median_training_X = training_X.fillna(training_X.median())

# training_X.to_excel('training_X_to_impute.xlsx', index = False)

# this is the code I ran in R to impute KNN
# library(readxl)
# X_to_impute <- read_excel("training_X_to_impute.xlsx")
# library(bnstruct)
# KNN_X = knn.impute(as.matrix(X_to_impute), k= 10)
# library(writexl)
# write_xlsx(as.data.frame(KNN_X), "~/Desktop/SIOP Analysis/training_X_imputed.xlsx")

url_KNN_X = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/datasets/training_X_imputed.xlsx'

KNN_training_X= pd.read_excel(url_KNN_X)


# ## Using 5-fold cross-validation on multiple models (5 Logistic Regression, 1 Naive Bayes, 1 Decision Tree, 7 Random Forests, 1 Support Vector Machine, 48 Neural Networks, 3 KNN models, and 243 XGBoost), find the best performing model to predict the "Hire" variable across imputations according to the AUC of the models.

# In[81]:


# hire_df = ordered_results[['UNIQUE_ID','Hire']]
# training_with_hire = pd.merge(training, hire_df, how='left', on='UNIQUE_ID')
# training_with_hire.to_excel('training_with_hire.xlsx', index= False)

# can also start here by importing training with hire, but you'll still need to go one block above to impute training_X
url_training_with_hire = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/datasets/training_with_hire.xlsx'

training_with_hire= pd.read_excel(url_training_with_hire) 


# In[82]:


# run models to predict Hire with median-, mean-, and KNN-imputed data
# training_y = training_with_hire.loc[:,"Hire"]

# mean_training= cross_validated_models(mean_training_X, training_y, 'Mean')
# median_training= cross_validated_models(median_training_X, training_y, 'Median')
# KNN_training= cross_validated_models(KNN_training_X, training_y, 'KNN')

# find and save best model from results
# training_results = pd.concat([mean_training,median_training,KNN_training], ignore_index=True)
# training_results.sort_values(by=['ROC_AUC'], ascending= False).head(1)
# training_results.to_excel('training_results.xlsx', index = False)

# best model for Hire: 
# median-imputed Neural Network with hidden_layer_sizes = (120,), activation= 'logistic', solver: 'adam' (mean ROC AUC of 0.865)

url_training_results = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/models/training_results.xlsx'

training_results= pd.read_excel(url_training_results)


# ### AUC visualization for best model for Hire

# In[84]:


# run the best model of: median-imputed Neural Network with hidden_layer_sizes = (120,), activation= 'logistic', solver: 'adam' (mean ROC AUC of 0.865)

def best_modelhire(X, y, X_imputation):
    
    X = X.apply(zscore)

    df_besthire = pd.DataFrame(columns= ['Model','Parameters', 'Accuracy', 'Recall','ROC_AUC'])

    score_besthire = pd.DataFrame(cross_validate(MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "adam", max_iter= 1000,random_state=0), X, y, scoring = ['recall','accuracy','roc_auc'], cv=5))[['test_accuracy','test_recall','test_roc_auc']].mean()

    df_besthire = df_besthire.append({'Model': "Neural Network",'Parameters': 'n_neighbors: '+str(120,), 'Accuracy': score_besthire[0],'Recall': score_besthire[1], 'ROC_AUC': score_besthire[2]}, ignore_index = True)
    
    return df_besthire

training_y = training_with_hire.loc[:,"Hire"]

best_modelhire(median_training_X,training_y, 'Median')


# In[85]:


# plot the AUC for the best model

training_y = training_with_hire.loc[:,"Hire"]

X = median_training_X.values
Y = training_y.values

cv = StratifiedKFold(n_splits=5)
classifier = MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "adam", max_iter= 1000,random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X, Y):
    probas_ = classifier.fit(X[train], Y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC of Neural Network',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()


# ## Import development dataset. Impute missing values according to the imputation of the best performing model. Predict the probability that each of these applicants is a Hire in a column called "Probability_of_Hire". Organize these probabilities in descending order. Create a new variable "Hire" which is 1 (hire) for the top half (a development dataset of 2250 total applicants requires that 1125 applicants be hired) and 0 (dont hire) for the bottom half

# In[102]:


# import non-imputed development data

url_participant_dev = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/datasets/participant_dev.xlsx'
dev = pd.read_excel(url_participant_dev, na_values= ' ').drop('split', axis=1)

# impute features of development data with medin
dev_X = dev.loc[:, 'SJ_Most_1':'PScale13_Q5']
median_dev_X = dev_X.fillna(dev_X.median())


# In[103]:


# train the top performing model and predict probabilities for training dataset
training_y = training_with_hire.loc[:,"Hire"]
classifier = MLPClassifier(hidden_layer_sizes = (120,), activation = 'logistic', solver= 'adam')
trained_classifier = classifier.fit(median_training_X, training_y)
dev_predicted_proba = trained_classifier.predict_proba(median_dev_X)


# In[104]:


dev['Probability_of_Hire'] = pd.DataFrame(dev_predicted_proba)[1]
dev_final = dev.sort_values(by=['Probability_of_Hire'], ascending = False)[['UNIQUE_ID','Probability_of_Hire']].reset_index()


# In[105]:


#  create a column called 'Hire' and fill it with 1s for the top 1125 instances
dev_final['Hire']= [1 if row < 1125 else 0 for row in range(len(dev_final))]
final= dev_final[['UNIQUE_ID','Hire']]


# ## Submit a dataframe with just "UNIQUE_ID" and "Hire". Submit it to the competition organizers as a .csv file and post the prediction metrics here

# In[ ]:


# final.to_csv('final_strategy_1.csv', index = False)


# In[ ]:


# Results from organizers:
# {"Percentage_of_true_top_performers_hired": 0.457, 
#  "Percentage_of_true_retained_hired": 0.572, 
#  "Percentage_of_true_retained_top_performers_hired": 0.455, 
#  "Adverse_impact_ratio": 0.9771, 
#  "Final_score": 46.1747}


# # Strategy 2: Use the top performing models for the training dataset to predict the probabilities of "Retained High Performer" in the development dataset. Then use those probabilities to determine who should be hired.

# ## Using 5-fold cross-validation on multiple models (5 Logistic Regression, 1 Naive Bayes, 1 Decision Tree, 7 Random Forests, 1 Support Vector Machine, 48 Neural Networks, 243 KNN models, and 1 XGBoost), find the best performing model to predict the "Protected_Group" variable in the training dataset across imputations according to the AUC of the models.

# In[113]:


# run models to predict Hire with median-, mean-, and KNN-imputed data

# training_y = training_with_hire.loc[:,"Protected_Group"]

# mean_training= cross_validated_models(mean_training_X, training_y, 'Mean')
# median_training= cross_validated_models(median_training_X, training_y, 'Median')
# KNN_training= cross_validated_models(KNN_training_X, training_y, 'KNN')

# find and save the best performing model
# training_protected_results = pd.concat([mean_training,median_training,KNN_training], ignore_index=True)
# training_protected_results.sort_values(by=['ROC_AUC'], ascending= False).head(1)
# training_protected_results.to_excel('training_protected_results.xlsx', index = False)

# best model for non_protected: 
# KNN-imputed Random Forest with n_estimators = 150 (mean OC AUC of 0.709)

url_training_protected_results = 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/models/training_protected_results.xlsx'

training_protected_results= pd.read_excel(url_training_protected_results)


# ### AUC visualization for best model for Hire

# In[115]:


# run the best model of: KNN-imputed Random Forest with n_estimators = 150 (mean OC AUC of 0.709)
def randomforest1(X, y, X_imputation):
    
    X = X.apply(zscore)

    rf1 = pd.DataFrame(columns= ['Model','n_estimators', 'Accuracy', 'Recall','ROC_AUC'])
    
    model = "Random Forest"
    
    estimators = 150
    
    rfscore = pd.DataFrame(cross_validate(rfc(n_estimators=150, random_state=0), X, y, scoring = ['recall','accuracy','roc_auc']))[['test_accuracy','test_recall','test_roc_auc']].mean()
    rf1 = rf1.append({'Model': model,'n_estimators': 150, 'Accuracy': rfscore[0],'Recall': rfscore[1], 'ROC_AUC': rfscore[2]}, ignore_index = True)

    return rf1

training_y = training_with_hire.loc[:,"Protected_Group"]
randomforest1(KNN_training_X,training_y, 'KNN')


# In[114]:


# plot the AUC for the best model

training_y = training_with_hire.loc[:,"Hire"]

X = KNN_training_X.values
Y = training_y.values

cv = StratifiedKFold()
classifier = rfc(n_estimators=1500, max_features="log2", random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X, Y):
    probas_ = classifier.fit(X[train], Y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC of Random Forest',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()


# ## Import development dataset and predict "Protected_Group" based on top performing model

# In[121]:


training_y = training_with_hire.loc[:,"Protected_Group"]
KNN_dev_protected_group = rfc(n_estimators= 150).fit(KNN_training_X.apply(zscore), training_y).predict(KNN_dev_X.apply(zscore))
dev['Predicted_Protected_Group'] = KNN_dev_protected_group


# ## Split development dataset based on predicted "Protected_Group"

# In[122]:


dev_protected = dev[dev['Predicted_Protected_Group']==1].reset_index()
dev_non_protected = dev[dev['Predicted_Protected_Group']==0].reset_index()


# ## PROTECTED: Impute and predict probabilities of being "Retained High Performer" for the Protected development group based on the best performing model for the Protected training group.

# In[123]:


# best model for non_protected: mean-imputed Neural Network with hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd"

# impute features of development protected data with KNN
dev_protected_X = dev_protected.loc[:, 'SJ_Most_1':'PScale13_Q5']

mean_dev_protected_X = dev_protected_X.fillna(dev_protected_X.mean())


# In[124]:


# train the top performing model (with x normalized) and predict probabilities for training dataset (with x normalized)
protected_y = protected.loc[:,"Retained_High_Performer"]

dev_protected['Probability_of_Retained_High_Performer']= pd.DataFrame(MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd").fit(mean_protected_X.apply(zscore), protected_y).predict_proba(mean_dev_protected_X.apply(zscore)))[1]


# ## NON-PROTECTED: Impute and predict probabilities of being "Retained High Performer" for the Non-Protected development group based on the best performing model for the Non-Protected training group.

# In[125]:


# best model for non_protected: KNN-imputed Neural Network with hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd"

dev_non_protected_X = dev_non_protected.loc[:, 'SJ_Most_1':'PScale13_Q5']

# dev_non_protected_X.to_excel('dev_non_protected_X_to_impute.xlsx', index = False)

# this is the code I ran in R to impute KNN
# library(readxl)
# X_to_impute <- read_excel('dev_non_protected_X_to_impute.xlsx')
# library(bnstruct)
# KNN_X = knn.impute(as.matrix(X_to_impute), k= 10)
# library(writexl)
# write_xlsx(as.data.frame(KNN_X), "~/Desktop/SIOP Analysis/dev_non_protected_X_imputed.xlsx')

url_dev_non_protected_X= 'https://raw.github.com/mneedle/ML-Greedy-Search-in-Python/master/datasets/dev_non_protected_X_imputed.xlsx'

KNN_dev_non_protected_X= pd.read_excel(url_dev_non_protected_X)


# In[126]:


# train the top performing model (with x normalized) and predict probabilities for training dataset (with x normalized)
non_protected_y = non_protected.loc[:,"Retained_High_Performer"]

dev_non_protected_predict_proba = MLPClassifier(hidden_layer_sizes = (120,), activation = "logistic", solver = "sgd").fit(KNN_non_protected_X.apply(zscore), non_protected_y).predict_proba(KNN_dev_non_protected_X.apply(zscore)) 

dev_non_protected['Probability_of_Retained_High_Performer']= pd.DataFrame(dev_non_protected_predict_proba)[1]


# ## Organize the Protected development probabilities in ascending order and concatenate them onto the Non-Protected development probabilities organized in descending order.
# 
# 

# In[127]:


columns = ['UNIQUE_ID','Predicted_Protected_Group','Probability_of_Retained_High_Performer']

dev_protected_ascending = dev_protected.sort_values(by=['Probability_of_Retained_High_Performer'], ascending = True).loc[:,columns]
dev_non_protected_descending = dev_non_protected.sort_values(by=['Probability_of_Retained_High_Performer'], ascending = False).loc[:,columns]


# In[128]:


ordered_results = pd.concat([dev_protected_ascending,dev_non_protected_descending], ignore_index=True).reset_index()


# ## Hire proportional numbers of development Protected and Non-Protected applicants based on how many Protected (416) and Non-Protected (1834) were predicted

# In[129]:


# create dataframe with column for unfairness
dev_protected_total = len(dev_protected)
dev_protected_list = list(ordered_results['Predicted_Protected_Group'])
unfairness_list = [] 
for i in range(len(ordered_results)-1125):
    dev_protected_identified = sum(dev_protected_list[i:i+1125])
    dev_protected_hire_rate = dev_protected_identified/dev_protected_total
    dev_non_protected_hire_rate = (1125-dev_protected_identified)/(2250-dev_protected_total)
    unfairness = abs(1-dev_protected_hire_rate/dev_non_protected_hire_rate)
    unfairness_list += [unfairness]

output = pd.DataFrame({'Unfairness':unfairness_list})


# In[130]:


# create new column for total probability of retained_high_performer
retained_high_performer_list = list(ordered_results['Probability_of_Retained_High_Performer'])
retained_high_performer_identified_list = [] 
for i in range(len(ordered_results)-1125):
    retained_high_performer_identified = sum(retained_high_performer_list[i:i+1125])
    retained_high_performer_identified_list += [retained_high_performer_identified]

output['Total_Probability_of_Retained_High_Performer']=retained_high_performer_identified_list


# In[131]:


output['Score']= output['Total_Probability_of_Retained_High_Performer']-abs(output['Unfairness'])*output['Total_Probability_of_Retained_High_Performer']


# ## Find the combination that results in the maximum final score and create a new "Hire" variable denoting that this half of applicants should be hired and the other half should not

# In[132]:


max_index = output[output['Score']==output['Score'].max()].index


# In[133]:


#  create a column called 'Hire' and fill it with 1s for the 1125 included in the max Score distribution

hire=[]
for i in range(len(ordered_results)):
    if ((i >= max_index) & (i < (max_index+1125))):
        hire += [1]
    else:
        hire += [0]

ordered_results['Hire'] = hire


# In[134]:


ordered_results['Hire'].value_counts()


# ## Submit a dataframe with just "UNIQUE_ID" and "Hire". Submit it to the competition organizers and post the prediction metrics here.

# In[135]:


final= ordered_results[['UNIQUE_ID','Hire']]
# final.to_csv('final_strategy_2.csv', index = False)


# In[136]:


# results from organizers:
# {"Percentage_of_true_top_performers_hired": 0.613, 
#  "Percentage_of_true_retained_hired": 0.566, 
#  "Percentage_of_true_retained_top_performers_hired": 0.605, 
#  "Adverse_impact_ratio": 1.0312, 
#  "Final_score": 56.5861}

