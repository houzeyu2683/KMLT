

##
##  Packages.
from torch.utils.data import DataLoader


##
##  The [loader] class.
class loader:

    def __init__(self, train=None, check=None, test=None, batch=32):

        if(train):
            
            self.train = DataLoader(train, batch_size=batch, shuffle=True , drop_last=False)
            pass

        if(check):

            self.check  = DataLoader(check , batch_size=batch, shuffle=False, drop_last=False)
            pass

        if(test):

            self.test  = DataLoader(test , batch_size=batch, shuffle=False, drop_last=False)
            pass
    
    def available(self, which='train'):

        if(which=='train'):

            try:

                next(iter(self.train))
                return(True)

            except:

                return(False)

        if(which=='check'):

            try:

                next(iter(self.check))
                return(True)

            except:

                return(False)

        if(which=='test'):

            try:

                next(iter(self.test))
                return(True)

            except:

                return(False)

