

##
##  Packages.
import torch, os


##
##  The [dataset] class.
class dataset(torch.utils.data.Dataset):

    def __init__(self, table, target=None, variable=None, image=None, text=None):

        self.table    = table
        pass

        self.target   = target
        self.variable = variable
        self.image    = image
        self.text     = text
        pass
    
    def __len__(self):

        length = len(self.table)
        return(length)

    def __getitem__(self, index):

        item  = self.table.iloc[index, :]
        batch = {"target":[], "variable":[], "image":[], "text":[]}
        pass

        if(self.variable):

            batch['variable'] = self.variable(item)
            pass
        
        if(self.image):

            batch['image'] = self.image(item)
            pass

        if(self.text):

            batch['text'] = self.text(item)
            pass

        if(self.target):

            batch['target'] = self.target(item)
            pass

        return(batch)

