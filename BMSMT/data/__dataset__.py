

##
##  Packages.
import torch, PIL.Image


##
##  Class for dataset.
class dataset(torch.utils.data.Dataset):

    def __init__(self, table, variable=None, image=None, text=None):

        self.table    = table
        pass
        
        ##  Define the process function for each type, case by case.
        self.variable = variable
        self.image    = image
        self.text     = text
        pass
    
    def __len__(self):

        length = len(self.table)
        return(length)

    def __getitem__(self, index):

        ##  Handle batch, case by case.
        batch = {"image":None, "text": None}
        
        ##  Select item by index.
        item  = self.table.iloc[index, :]

        ##  Process variable, case by case.
        if(self.variable):

            pass
        
        ##  Process image, case by case.
        if(self.image):

            batch['image'] = self.image(PIL.Image.open(item['image']).convert("RGB"))
            pass
        
        ##  Process text, case by case.
        if(self.text):

            batch['text'] = self.text(item['label'])
            pass
        
        ##  Handle outout, case by case.
        output = batch['image'], batch['text']
        return(output)

