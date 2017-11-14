import csv,sys
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve
data=[]

def Calc_equal_err_rate(realUser_scores,fakeUser_scores,labels):
    
#     print ("Calculating equal error rate for subject::", subject)
    
    # fpr = ( array ) Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
    # tpr = ( array ) Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
    # thresholds = ( array ) Decreasing thresholds on the decision function used to compute fpr and tpr. 
    fpr, tpr, thresholds = roc_curve(labels, realUser_scores + fakeUser_scores)
   
    # The hit rate is the frequency with which impostors are detected (i.e. tpr = 1 − miss rate)
    # distance between false and missed alarm rates
    all_dist = (1 - tpr) - fpr
    
    # index where min and max value in dists array
    idx1 = np.argmin(all_dist[all_dist >=0 ])
    idx2 = np.argmax(all_dist[all_dist < 0])
    
     # equal_error_rate calculation
    false_alarm_rates = fpr #(false positive rate) 
    miss_rate = (1 - tpr)
    a = miss_rate[idx1]
    b = false_alarm_rates[idx1]
    x_ptr = [a, b]
    
    a = miss_rate[idx2]
    b = false_alarm_rates[idx2]
    y_ptr = [a, b]
    
    num =  x_ptr[0] -x_ptr[1]
    denom = y_ptr[1] - x_ptr[1] - y_ptr[0] + x_ptr[0]
    m = num  / denom
   
    # using line equation 
    equal_err_rate = x_ptr[0] + m * ( y_ptr[0] - x_ptr[0] )
#     print (round(equal_err_rate,4))
    
    return equal_err_rate
    

def SVM(person):
    # Consider current subject as real and rest as fake
    realUser_data = data.loc[data.subject == person, "H.period":"H.Return"]
    
    if (len(realUser_data) == 0):
        print ("No data found for the given user")
        return 0
    
    real_train = np.array((realUser_data[:200]).values)
    
    # True test set (200 records)
    real_test = np.array((realUser_data[200:]).values)

    fakeUser_data = data.loc[data.subject != person, :]
    
    # False set (250 records, 5 per fake user, 50 fake users in all)
    fake_test = np.array((fakeUser_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]).values)

    clf = OneClassSVM(kernel='rbf',gamma=26)
    clf.fit(real_train)
    
    realUser_scores = []                         # real user score
    fakeUser_scores = []                         # imposter user score
    
    # Calculate score for real user test data
    realUser_scores =list( -clf.decision_function(real_test))
    
    # Calculate score for fake user test data
    fakeUser_scores = list(-clf.decision_function(fake_test))
    
    # true label
    labels = [0]*len(realUser_scores) + [1]*len(fakeUser_scores)

    Equal_err_rate = Calc_equal_err_rate(realUser_scores,fakeUser_scores,labels)
    print ("Equal err rate:: ",Equal_err_rate)
    
    return Equal_err_rate


def main():
    global data
    data = pd.read_csv("DSL-StrongPasswordData.csv")
    
    # find all unique users
    subjects = data["subject"].unique()
    print ("Number of unique users:: ",len(subjects))
    
    header = [c for c in data]
    print ("Dataset contains fields : ")
    for h in header :
        print (h, end=', ')
    print ()
    equal_err_rate =[]
    
    for person in subjects:
        equal_err_rate.append(SVM(person))
    
    Mean = np.mean(equal_err_rate)
    StdDev = np.std(equal_err_rate)
    print ("Mean:: ", Mean, "StdDev :: ", StdDev)


if __name__ == main():
    main()

