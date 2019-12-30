import requests
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score,f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import pandas as pd

def getNumericInfo(recordName):
    # get patient id
    # get list of parameters
    # get record number
    # get sampling rate
    # get duration
    # get data time stamp
    urlhea = 'https://archive.physionet.org/physiobank/database/mimic3wdb/matched/'
    urlhea = urlhea + recordName + '.hea'
    heareq = requests.get(urlhea, verify=False)
    recordNameArr = recordName.split("/")
    patientId = recordNameArr[1]
    patientId = int(patientId[1:])
    lineCount = 0
    paramList = []
    for line in heareq.iter_lines():
        if(lineCount==0):
            firstline = line.decode('utf-8').split(" ")
            samples = int(firstline[3])
            samplingrate = int(float((firstline[2]).split('/')[0])*60)
            hours_duration = float(samples)/(60*float(samplingrate))
            time = str(firstline[4])
            date = str(firstline[5])
            date_time = time + ' ' + date
        if(lineCount==1):
            secondLine = line.decode('utf-8').split(" ")
            recNum = int(secondLine[0][:-5])
            param = secondLine[8:]
            param = "".join(param)
            paramList.append(param)
        if(lineCount>1):
            nthLine = line.decode('utf-8').split(" ")
            param = nthLine[8:]
            param = "".join(param)
            paramList.append(param)
        lineCount+=1
    return patientId, recNum, samples, samplingrate, hours_duration, date_time, paramList


# function to print confusion matrix and classification metrics
def metrics_analysis(X_test,y_test,model):
    # Predicting the Test set results
    y_pred = model.predict(X_test)
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))

# function to plot auc curve
def plot_rocauc(X_test,y_test,model):
    prob = model.predict_proba(X_test)[:,1]
    logit_roc_auc = roc_auc_score(y_test, prob)
    fpr, tpr, thresholds = roc_curve(y_test, prob)
    plt.figure()
    plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('Log_ROC')
    plt.show()

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    display(optimal_threshold)
    return optimal_threshold

# function to print classification metrics using optmial threshold
def display_classification_metrics(X_test,y_test,model,THRESHOLD):
    prob = model.predict_proba(X_test)[:,1]
    preds = np.where(prob > THRESHOLD, 1, 0)
    print(classification_report(y_test, preds))
    df=pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                   precision_score(y_test, preds),f1_score(y_test,preds), roc_auc_score(y_test, prob)], 
             index=["accuracy", "recall", "precision", "f1-score","roc_auc_score"],columns=['Value'])
    display(df)