

##
##  Packages.
import torch


##
##  Class for dataset.
class dataset(torch.utils.data.Dataset):

    def __init__(self, table, image=None, text=None):

        self.table    = table
        pass
        
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
        
        ##  Process image, case by case.
        if(self.image):

            batch['image'] = self.image(item['image'])
            pass
        
        ##  Process text, case by case.
        if(self.text):

            batch['text'] = self.text(item['label'])
            pass
        
        ##  Handle outout, case by case.
        output = batch['image'], batch['text']
        return(output)

