# Comparing Anomaly-Detection Algorithms for Keystroke Dynamics

### Objective
- To collect a keystroke-dynamics data set
- To develop a repeatable evaluation procedure
- To measure the performance of a range of detectors (results can be compared soundly).

### Motivation
- To reduce the security threats from external attacker and insiders.
- Using some methods, to discriminate between the genuine user of an account and an impostor, i.e.  “Anomaly-Detection Task” .

### Proposed approach
Keystroke dynamics or typing dynamics refers to the automated method of identifying or confirming the identity of an individual based on the manner and the rhythm of typing on a keyboard. Keystroke dynamics is a behavioral biometric, this means that the biometric factor is ‘something you do’.


Possible applications of keystroke dynamics include acting as an electronic fingerprint, or in an access-control mechanism. A digital fingerprint would tie a person to a computer-based crime in the same manner that a physical fingerprint ties a person to the scene of a physical crime. Access control could incorporate keystroke dynamics both by requiring a legitimate user to type a password with the correct rhythm, and by continually authenticating that user while they type on the keyboard.

### Dataset
Now to collect dataset author proposed an experimentation, where we recruit 51 subjects (typist) who type same password, and each subject typed the password 400 times over 8 sessions (50 repetitions per session).  They waited at least one day between sessions, to capture some of the day-to-day variation of each subject's typing. The password .tie5Roanl was chosen to be representative of a strong 10-character password.

### Applications

- Acting as an electronic fingerprint
- Access-control and authentication mechanism
- Detecting computer-based crimes

### Existing work
Comparing Anomaly-Detection Algorithms for Keystroke Dynamics
- http://www.cs.cmu.edu/~maxion/pubs/KillourhyMaxion09.pdf
- http://www.cs.cmu.edu/~keystroke/


### Detectors implemented and Results

| Detector                       | Average Equal-Error Rate | Standard deviation of EER |
|--------------------------------|--------------------------|---------------------------|
| Manhattan Scaled Detector      | 0.0945                   | 0.068375                  |
| Outlier Count (z-score)        | 0.103167                 | 0.07691                   |
| Nearest Neighbor (Mahalanobis) | 0.1075                   | 0.06213                   |
| SVM (one-class)                | 0.12068                  | 0.0586                    |
| Manhattan Filtered             | 0.12535                  | 0.081299                  |
| Mahalanobis                    | 0.1337                   | 0.06678                   |
| Mahalanobis Normed             | 0.1337                   | 0.06678                   |
| Manhattan                      | 0.15                     | 0.09                      |
| K-Means                        | 0.1559                   | 0.072                     |
| Neural Network (auto-assoc)    | 0.16417                  | 0.0914199                 |
| Euclidean                      | 0.16929                  | 0.0931429                 |
| Euclidean Normed               | 0.2107                   | 0.1174                    |
| Neural Network (standard)      | 0.6551                   | 0.1866                    |

### Files for detectors

- Manhattan (scaled) - Manhattan_Scaled.ipynb
- SVM - SVM.ipynb
- Euclidean - Euclidean.ipynb
- Mahanlobis - KeystrokeDynamics.ipynb
