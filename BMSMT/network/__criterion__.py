

##
##  Packages.
import torch


##
##  Class for criterion.
class criterion:

    ##  Cross entropy loss.
    def cel(weight=None):

        loss   = torch.nn.CrossEntropyLoss(ignore_index=65)
        pass
        # ##  Classification weight.
        # if(weight):
            
        #     weight = torch.tensor(weight, dtype=torch.float32)
        #     loss   = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=65)
        #     pass

        # else:

        #     loss = torch.nn.CrossEntropyLoss()
        #     pass

        return(loss)

    ##  Mean absolute error.
    def mae():

        loss = torch.nn.L1Loss()
        return(loss)



