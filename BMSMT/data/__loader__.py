


##
##  Packages.
import torch, pickle
from torch.utils.data import DataLoader


##
##
path = "SOURCE/PICKLE/VOCABULARY.pickle"
with open(path, 'rb') as paper:

    vocabulary = pickle.load(paper)
    pass


##
##


##
##
def sample(collection):
    batch = {
        "image":[],
        "text":[]
    }
    for index, (image, text) in enumerate(collection):

        image = torch.unsqueeze(image, dim=0)
        batch['image'].append(image)
        pass

        text = [vocabulary['<bos>']] + [vocabulary[i] for i in text] + [vocabulary['<eos>']]
        text = torch.tensor(text, dtype=torch.long)
        batch['text'].append(text)
        pass

    batch['text']  = torch.nn.utils.rnn.pad_sequence(batch['text'], padding_value=vocabulary['<pad>'])
    batch['image'] = torch.cat(batch['image'], dim=0)
    batch['size'] = index+1
    return(batch['image'], batch['text'], batch['size'])



##
##  Class for loader of dataset.
class loader:

    def __init__(self, train=None, check=None, test=None, batch=32):

        if(train):
            
            self.train = DataLoader(train, batch_size=batch, shuffle=True , drop_last=False, collate_fn=sample)
            pass

        if(check):

            self.check  = DataLoader(check , batch_size=batch, shuffle=False, drop_last=False, collate_fn=sample)
            pass

        if(test):

            self.test  = DataLoader(test , batch_size=batch, shuffle=False, drop_last=False, collate_fn=sample)
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

