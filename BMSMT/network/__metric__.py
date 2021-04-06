

##
##  Packages.
import numpy
import pandas
import sklearn
from sklearn import metrics


##
##  The [metric] class.
class metric:

    ##  Mean absolute error.
    def mae(target, likelihood):

        score = metrics.mean_absolute_error(y_true=target, y_pred=likelihood)
        return(score)

    ##  Area under curve.
    def auc(target, likelihood):

        score = metrics.roc_auc_score(y_true=target, y_score=likelihood)
        return(score)

    ##  Accuracy.
    def accuracy(target, prediction):

        score = metrics.accuracy_score(y_true=target, y_pred=prediction)
        return(score)

    ##  Entropy.
    def entropy(target, likelihood):
        
        score = metrics.log_loss(y_true=target, y_pred=likelihood)
        return(score)

    ##  Confusion.
    def confusion(target, prediction):
        
        table = metrics.confusion_matrix(y_true=target, y_pred=prediction)
        score = str(table.ravel().tolist())
        return(score)

