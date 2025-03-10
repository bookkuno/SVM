Train SVM (UDP dataset)
- 1)start with the simplest/base case: linear svm
   > use only 1 dataset (UDP) for train and test
   > do somedata processing
   > got a accuracy ~ 99%
- 2)then use two datasets (one for train and validate); another for test | but still linear svm (try others but no improvment to the accuracy)
   > got low accuracy ~ 10%
       > potential problems: 
            1)Imbalanced Classes:
              The test set has a severe imbalance:
              Class 0 and 1 dominate, while Class 2 has no samples.
              The model struggles with rare classes.
            2)Overfitting: The model performs well on the validation set but poorly on the test set, indicating overfitting to the training/validation data.
            Dataset Discrepancies:
            3)Possible differences between training/validation and test datasets in terms of class distribution or feature scaling.
      > identifying the problem(s): Class Imbalance: Class 0 and 1 dominate, while Class 2 has no samples in the test set. This imbalance skews model performance.
      > solution(s): add class_weight='balanced' to the line: svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced')
         - accuracy increased by 5% from 10% to 15%

Train SVM (IDS2018 dataset)
- 1) Same as 1 in UDP
- 2) same setting as the one on UDP (after adding class_weight='balanced')
   > SVM_Type, Train_Validate_Set, Train_Sampling_Size(0-1 = 0-100%),  Test_Set, Test__Sampling_Size(0-1 = 0-100%), Accuracy
   -----------DDoS datasets(random_state = 42)-------------- https://www.unb.ca/cic/datasets/ids-2018.html
   > Train with DoS attacks-GoldenEye(15) 375MB, 80 columns, Benign 95% | DoS attacks-GoldenEye 4% | Other 1%
      https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv?resource=download&select=02-15-2018.csv
      🔵15 vs 20
      - ✅linear, 15, 0.1, 20, 0.01, 0.919 | Best Parameters: {'C': 100, 'kernel': 'linear'}
      - poly, 15, 0.1, 20, 0.01, 0. | 
      - rbf, 15, 0.1, 20, 0.01, 0. | 
      🔵15 vs 21
      - ✅linear, 15, 0.1, 21, 0.1, 0.393 | Best Parameters: {'C': 0.1, 'kernel': 'linear'}
      - ✅poly, 15, 0.1, 21, 0.1, 0.342 | Best Parameters: {'C': 10, 'kernel': 'poly'} 
      - ✅rbf, 15, 0.1, 21, 0.1, 0.342 | Best Parameters: {'C': 100, 'kernel': 'rbf'}
   > Train with DDOS attack-LOIC-HTTP(20) 4GB, 84 columns, Benign 93% | DDoS attacks-LOIC-HTTP 7%
      https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv?select=02-20-2018.csv
      🟡20 vs 21
      - ✅linear, 20, 0.01, 21, 0.1, 0.827 | Best Parameters: {'C': 100, 'kernel': 'linear'}
      - ✅poly, 20, 0.01, 21, 0.1, 0.838 | Best Parameters: {'C': 100, 'kernel': 'poly'}
      - ✅rbf, 20, 0.01, 21, 0.1, 0.723 | Best Parameters: {'C': 100, 'kernel': 'rbf'}
      🟡20 vs 15
      - ✅linear, 20, 0.01, 15, 0.1, 0.942 | Best Parameters: {'C': 100, 'kernel': 'linear'}
      - ✅poly, 20, 0.01, 15, 0.1, 0.959 | Best Parameters: {'C': 100, 'kernel': 'poly'}
      - ✅rbf, 20, 0.01, 15, 0.1, 0.961 | Best Parameters: {'C': 100, 'kernel': 'rbf'}
   > Train with DDoS attacks-HOIC(21) 329MB (10 times samller than 20), 80 columns(no first 4 columns in 20: Flow ID, Src IP, Src Port, Dst IP), Benign 34% | DDOS attack-HOIC 65% | Other 0%
      https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv/data?select=02-21-2018.csv
      🔴21 vs 15
      - ✅linear, 21, 0.1, 15, 0.1, 0.307 | Best Parameters: {'C': 0.1, 'kernel': 'linear'}
      - ✅poly, 21, 0.1, 15, 0.1, 0.201 | Best Parameters: {'C': 10, 'kernel': 'poly'}
      - ✅rbf, 21, 0.1, 15, 0.1, 0.467 | Best Parameters: {'C': 10, 'kernel': 'rbf'}
      🔴21 vs 20
      - ✅linear, 21, 0.1, 20, 0.01, 0.393 | Best Parameters: {'C': 0.1, 'kernel': 'linear'}
      - ✅poly, 21, 0.01, 20, 0.1, 0.330 | Best Parameters: {'C': 100, 'kernel': 'poly'}
      - ✅rbf, 21, 0.01, 20, 0.1, 0.546 | Best Parameters: {'C': 10, 'kernel': 'rbf'}

   > Since each dataset has different label names for DDoS, have to check for their correlation/dependency -> possible to use it intercgangeably (rename to just DDoS for all)
      - Methods to check:
         - Chi-Square tests for categorical labels.
         - Train a model to predict B and test it on C (or vice versa).
