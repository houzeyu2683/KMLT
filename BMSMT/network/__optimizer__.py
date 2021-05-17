

##
##  Packages.
import torch


##
##  The [optimizer] class.
class optimizer:

    def sgd(model):

        output = torch.optim.SGD(
            model.parameters(), 
            lr=0.001, momentum=0.9,
            dampening=0, weight_decay=0.1, 
            nesterov=False
        )
        return(output)

    def adam(model):

        output = torch.optim.Adam(
            model.parameters(), 
            lr=0.0001, betas=(0.9, 0.999), 
            eps=1e-09, weight_decay=1e-5, amsgrad=False
        )
        return(output)

    # def scheduler(optimizer, step, gamma=0.1):
        
    #     output = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma, last_epoch=-1, verbose=False)
    #     return(output)