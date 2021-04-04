

##
##  Load the packages.
import numpy
import pandas
import sklearn
from sklearn import metrics


##
##  The [auc], area under curve function.
def auc(target, likelihood):

    score = metrics.roc_auc_score(y_true=target, y_score=likelihood)
    return(score)


##
##  The [accuracy] function.
def accuracy(target, prediction):

    score = metrics.accuracy_score(y_true=target, y_pred=prediction)
    return(score)


##
##  The [entropy] function.
def entropy(target, likelihood):
    
    score = metrics.log_loss(y_true=target, y_pred=likelihood)
    return(score)


##
##  The [confusion] function.
def confusion(target, prediction):
    
    table = metrics.confusion_matrix(y_true=target, y_pred=prediction)
    score = str(table.ravel().tolist())
    return(score)

