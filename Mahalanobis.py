import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from scipy.spatial.distance import cityblock, mahalanobis, euclidean

data = pd.read_csv("DSL-StrongPasswordData.csv")


class MahalanobisDetector:

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
        e = x[0] + a * ( y[0] - x[0] )

        self.equal_error_rate.append(e)



    def __init__(self, subjects):
        self.equal_error_rate = []
        self.mean_vector = []
        self.subjects = subjects

    
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            gen = self.test_genuine.iloc[i].values
            diff = gen - self.mean_vector
            A = np.dot(diff.T, self.covinv)
            self.user_scores.append(np.dot(A, diff))

        for i in range(self.test_imposter.shape[0]):
            gen = self.test_imposter.iloc[i].values
            diff = gen - self.mean_vector
            A =np.dot(diff.T, self.covinv)
            self.imposter_scores.append(np.dot(A, diff))

    def training(self):
        self.mean_vector = self.train.mean().values
        self.covinv = np.linalg.inv(np.cov(self.train.T))

    def generateTestTrain (self, genuine_user_data) :
        return genuine_user_data[:200], genuine_user_data[200:]

    def evaluate(self):

        for subject in self.subjects:

            self.user_scores = []
            self.imposter_scores = []

            # Consider current subject as genuine and rest as imposters
            imposter_data = data.loc[data.subject != subject, :]
            genuine_user_data = data.loc[data.subject == subject, "H.period":"H.Return"]

            # genuine user's first 200 time vectors for training
            self.train, self.test_genuine = self.generateTestTrain(genuine_user_data)

            self.training()

            # False set (250 records, 5 per imposter, 50 imposters in all)
            self.test_imposter = imposter_data.groupby("subject").head(5).loc[:, "H.period":"H.Return"]

            self.testing()

            self.evaluateEER(self.user_scores, self.imposter_scores)

# 51 total
subjects = data["subject"].unique()

det = MahalanobisDetector(subjects)
det.evaluate()
print (np.mean(det.equal_error_rate))
print (np.std(det.equal_error_rate))