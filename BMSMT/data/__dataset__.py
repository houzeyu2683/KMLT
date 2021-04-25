

##
##  Packages.
import torch, os, PIL.Image


##
##  Class for dataset.
class dataset(torch.utils.data.Dataset):

    def __init__(self, table, target=None, variable=None, image=None, text=None):

        self.table    = table
        pass
        
        ##  Define the process function for each type, case by case.
        self.target   = target
        self.variable = variable
        self.image    = image
        self.text     = text
        pass
    
    def __len__(self):

        length = len(self.table)
        return(length)

    def __getitem__(self, index):

        ##  Handle batch, case by case.
        batch = {"feature":[], "target":[]}
        
        ##  Select item by index.
        item  = self.table.iloc[index, :]

        ##  Process variable, case by case.
        if(self.variable):

            pass
        
        ##  Process image, case by case.
        if(self.image):

            midden = PIL.Image.open(item['image']).convert("RGB")
            batch['feature'] += [self.image(midden)]
            pass
        
        ##  Process text, case by case.
        if(self.text):

            pass
        
        ##  Process target, case by case.
        if(self.target):

            midden = item['InChI']
            batch['target'] += [self.target(midden)]
            pass
        
        ##  Handle outout, case by case.
        output = batch['feature'][0], batch['target'][0]
        return(output)

