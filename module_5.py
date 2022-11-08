#Module_5 
#Models 

#Import the necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = "{:,.2f}".format
pd.options.display.precision = 3



#Load the data set
data = pd.read_csv('/Volumes/GoogleDrive/My Drive/MIT/Digital Marketing Analytics/High Note Freemimum Case Study/High Note data.csv')
df = data.copy()


#Drop variables with NAN's
df = df.drop(['delta2_shouts'], axis= 1)
df = df.dropna()

#Separate periods in diferent data sets, to run models 

df_cur = df[['adopter','age', 'male', 'tenure', 'good_country', 'friend_cnt', 'avg_friend_age',
       'avg_friend_male', 'friend_country_cnt', 'subscriber_friend_cnt',
       'songsListened', 'lovedTracks', 'posts', 'playlists', 'shouts']]


df_pre = df[['adopter','age', 'male', 'tenure',
'delta1_friend_cnt', 'delta1_avg_friend_age', 'delta1_avg_friend_male',
'delta1_friend_country_cnt', 'delta1_subscriber_friend_cnt',
'delta1_songsListened', 'delta1_lovedTracks', 'delta1_posts',
'delta1_playlists', 'delta1_shouts','delta1_good_country']]


df_pos = df[['adopter', 'age', 'male', 'tenure', 'delta2_friend_cnt',
'delta2_avg_friend_age', 'delta2_avg_friend_male',
'delta2_friend_country_cnt', 'delta2_subscriber_friend_cnt',
'delta2_songsListened', 'delta2_lovedTracks', 'delta2_posts',
'delta2_playlists', 'delta2_good_country']]


#Separate dependent and independent variables from the data sets created 

Y_cur = df_cur['adopter']

X_cur = df_cur.drop(['adopter'], axis= 1)


Y_pre = df_pre['adopter']

X_pre = df_pre.drop(['adopter'], axis= 1)


Y_pos = df_pos['adopter']

X_pos = df_pos.drop(['adopter'], axis= 1)


#Scaling the data for the current data set.

sc=StandardScaler()
X_scaled=sc.fit_transform(X_cur)
X_scaled=pd.DataFrame(X_scaled, columns=X_cur.columns)

#splitting the data
x_train,x_test,y_train,y_test=train_test_split(X_scaled,Y_cur,test_size=0.3,random_state=1)


#creating metric function 
def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['Not Subscriber', 'Subscriber'], yticklabels=['Not Subscriber', 'Subscriber'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()



#Building a Logistic Regression Model 

lg=LogisticRegression()
lg.fit(x_train,y_train)

#checking the performance on the training data
y_pred_train = lg.predict(x_train)
metrics_score(y_train, y_pred_train)


#Correcting imbalance of classes with oversampling

df_majority = df_cur[(df_cur['adopter']== 0)]   
df_minority = df_cur[(df_cur['adopter']== 1)]   


# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,      # sample with replacement
                                 n_samples=40300)   # to match majority class


#Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

#Now our data set is balanced 
df_upsampled.adopter.value_counts()

#Splitting the data on the resampled data set

x = df_upsampled.drop('adopter', axis=1)
y = df_upsampled['adopter']

sc=StandardScaler()
X_scaled=sc.fit_transform(x)
X_scaled=pd.DataFrame(X_scaled, columns=x.columns)

#splitting the data
x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=1)


#Run the model again with resampled data
lg_re=LogisticRegression()
lg_re.fit(x_train,y_train)

#checking the performance on the training data
y_pred_train = lg_re.predict(x_train)
metrics_score(y_train, y_pred_train)

#checking the performance on the test dataset
y_pred_test = lg_re.predict(x_test)
metrics_score(y_test, y_pred_test)

#printing the coefficients of logistic regression
cols=x.columns
coef_lg_re=lg_re.coef_
pd.DataFrame(coef_lg_re,columns=cols).T.sort_values(by=0,ascending=False)



#Getting the odds of the model
odds = np.exp(lg_re.coef_[0]) #finding the odds
pd.DataFrame(odds, x_train.columns, columns=['odds']).sort_values(by='odds', ascending=False) 



y_scores_lg_re=lg_re.predict_proba(x_train) #predict_proba gives the probability of each observation belonging to each class
precisions_lg_re, recalls_lg_re, thresholds_lg_re = precision_recall_curve(y_train, y_scores_lg_re[:,1])


import matplotlib.ticker as plticker

loc = plticker.MultipleLocator(base=.5) # this locator puts ticks at regular intervals


#Plot values of precisions, recalls, and thresholds
plt.figure(figsize=(10,7))
plt.plot(thresholds_lg_re, precisions_lg_re[:-1], 'b--', label='precision')
plt.plot(thresholds_lg_re, recalls_lg_re[:-1], 'g--', label = 'recall')
plt.xlabel('Threshold')
plt.legend(loc='upper left')
plt.locator_params(axis='x', nbins=15)
plt.ylim([0,1])
plt.show()


optimal_threshold1=.46
y_pred_train = lg_re.predict_proba(x_train)
metrics_score(y_train, y_pred_train[:,1]>optimal_threshold1)

y_pred_test2 = lg.predict(x_test)
metrics_score(y_test, y_pred_test)

#printing the coefficients of logistic regression
cols=x.columns
coef_lg=lg.coef_
pd.DataFrame(coef_lg,columns=cols).T.sort_values(by=0,ascending=False)



#Getting the odds of the model
odds = np.exp(lg.coef_[0]) #finding the odds
pd.DataFrame(odds, x_train.columns, columns=['odds']).sort_values(by='odds', ascending=False) 

