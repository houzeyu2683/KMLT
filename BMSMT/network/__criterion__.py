
##
import torch

##
class criterion:

    def entropy(weight=None):

        if(weight):
            
            weight = torch.tensor(weight, dtype=torch.float32)
            loss   = torch.nn.CrossEntropyLoss(weight=weight)
            pass

        else:

            loss = torch.nn.CrossEntropyLoss()
            pass

        return(loss)
