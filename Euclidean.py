
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

class EuclideanDetector:

    def __init__(self, subjects):
        self.mean_vector = []
        self.subjects = subjects
    
    def testing(self):

        for i in range(self.test_imposter.shape[0]):
            self.imposter_scores.append(np.linalg.norm(self.test_imposter.iloc[i].values - self.mean_vector))
    
        for i in range(self.test_genuine.shape[0]):
            self.user_scores.append(np.linalg.norm(self.test_genuine.iloc[i].values - self.mean_vector))          

    def evaluateEER(self, user_scores, imposter_scores):
        
        # true label
        labels = [0]*len(user_scores) + [1]*len(imposter_scores)
        
        # fpr = ( array ) Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
        # tpr = ( array ) Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        # thresholds = ( array ) Decreasing thresholds on the decision function used to compute fpr and tpr. 
        fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)

        # The hit rate is the frequency with which impostors are detected (i.e. tpr = 1 − miss rate)
        miss_rates = 1 - tpr

        # false alarm rates
        false_alarm_rates = fpr
        
        # distance between false and missed alarm rates 
        dists = miss_rates - false_alarm_rates

        # index where min and max value in dists array
        idx1 = np.argmin(dists[dists >= 0])
        idx2 = np.argmax(dists[dists < 0])

        # equal_error_rate calculation
        x = [miss_rates[idx1], false_alarm_rates[idx1]]
        y = [miss_rates[idx2], false_alarm_rates[idx2]]
        a = ( x[0] - x[1] ) / ( y[1] - x[1] - y[0] + x[0] )
        equal_error_rate = x[0] + a * ( y[0] - x[0] )

        return equal_error_rate

        

    def training(self):
        self.mean_vector = self.train.mean().values
     

    def evaluate(self):
        equal_error_rates = []
        
        for subject in self.subjects:

            self.user_scores = []
            self.imposter_scores = []
    
            # Consider current subject as genuine and rest as imposters
            imposter_data = data.loc[data.subject != subject, :]
            genuine_user_data = data.loc[data.subject == subject, "H.period":"H.Return"]
    
    
            # True test set (200 records)
            self.test_genuine = genuine_user_data[200:]
            # genuine user's first 200 time vectors for training
            self.train = genuine_user_data[:200]
    
            # False set (250 records, 5 per imposter, 50 imposters in all)
            sub = imposter_data.groupby("subject")
            self.test_imposter = sub.head(5).loc[:, "H.period":"H.Return"]
            
            self.training()
            self.testing()
            
            equal_error_rates.append(self.evaluateEER(self.user_scores, self.imposter_scores))
            
        return np.mean(equal_error_rates), np.std(equal_error_rates)        

data = pd.read_csv("DSL-StrongPasswordData.csv")

subjects = data["subject"].unique()

det = EuclideanDetector(subjects)

print ("(Mean , StdDev) : ", det.evaluate())