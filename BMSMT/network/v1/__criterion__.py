

##
##  Packages.
import torch


##
##  The [criterion] class.
class criterion:

    ##  Cross entropy loss.
    def entropy(weight=None):

        ##  Classification weight.
        if(weight):
            
            weight = torch.tensor(weight, dtype=torch.float32)
            loss   = torch.nn.CrossEntropyLoss(weight=weight)
            pass

        else:

            loss = torch.nn.CrossEntropyLoss()
            pass

        return(loss)

    ##  Mean absolute error.
    def mae():

        loss = torch.nn.L1Loss()
        return(loss)
