

##
##  Packages.
import torch


##
##
class criterion:

    ##  Cross entropy loss.
    def cel(weight=None, ignore=-100):

        ##  Classification weight.
        if(weight):
            
            weight = torch.tensor(weight, dtype=torch.float32)
            loss   = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore)
            pass

        else:

            loss = torch.nn.CrossEntropyLoss(ignore_index=ignore)
            pass

        return(loss)

    ##  Mean absolute error.
    def mae():

        loss = torch.nn.L1Loss()
        return(loss)



