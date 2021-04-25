

##
##  Packages.
import torch, torchvision
import torch.nn as nn


##
##  Class for model, case by case.
class model(torch.nn.Module):

    def __init__(self):

        super(model, self).__init__()
        pass
        
        ##  Word number.
        word = 95

        ##  Sequence length.
        sequence = 512

        ##  Layer structure.
        layer = {
            "convolutional":torchvision.models.resnet18(True),
            "sequential":nn.ModuleDict({str(i):nn.Linear(1000, word) for i in range(sequence)}),
            "rnn":nn.GRU(word, word, 2),
            "attention":nn.TransformerEncoderLayer(d_model=word, nhead=1),
            "embedded":nn.Embedding(word, word),
            "probable":nn.Softmax(dim=2)
        }
        self.layer = nn.ModuleDict(layer)
        pass

    def forward(self, batch):

        ##  Handle batch.
        feature, target = batch
        midden = {"feature":None, "target":None}

        ##  Feature forward.
        midden['feature']    = self.layer['convolutional'](feature)
        midden['feature']    = [torch.unsqueeze(self.layer['sequential'][i](midden['feature']), 1) for i in self.layer['sequential']]
        midden['feature']    = torch.cat(midden['feature'], 1)
        midden['feature'], _ = self.layer['rnn'](midden['feature'])
        midden['feature']    = self.layer['attention'](midden['feature'])
        midden['feature']    = self.layer['probable'](midden['feature'])

        ##  Target forward.
        midden['target']    = self.layer['embedded'](target)
        midden['target']    = self.layer['probable'](midden['target'])

        ##  Handle output.
        output = midden['feature'], midden['target']
        return(output)

    def load(self, path):
        
        weight = torch.load(path, map_location='cpu')
        self.load_state_dict(weight)
        pass



# layer = {
#     "convolutional":torchvision.models.resnet18(True),
#     "sequential":nn.ModuleDict({str(i):nn.Linear(1000, 27) for i in range(512)}),
#     "rnn":nn.GRU(27, 27, 2),
#     "attention":nn.TransformerEncoderLayer(d_model=27, nhead=3),
#     "embedded":nn.Embedding(27, 27),
#     "probable":nn.Softmax(dim=2)
# }
# layer = nn.ModuleDict(layer)



# midden = {"feature":None, "target":None}

# ##  圖片層====>文字機率
# midden['feature']    = layer['convolutional'](feature)
# midden['feature']    = [torch.unsqueeze(layer['sequential'][i](midden['feature']), 1) for i in layer['sequential']]
# midden['feature']    = torch.cat(midden['feature'], 1)
# midden['feature'], _ = layer['rnn'](midden['feature'])
# midden['feature']    = layer['attention'](midden['feature'])
# midden['feature']    = layer['probable'](midden['feature'])

# ##  文字===>編碼
# midden['target']    = layer['embedded'](target)
# midden['target']    = layer['probable'](midden['target'])







##
##
# feature = torch.randn((16, 3, 224, 226))
# target  = torch.randint(0, 27, (16, 512))
# batch = feature, target

# test = model()
# x, y = test(batch)
# x.shape
# y.shape
