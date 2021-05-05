

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
        
        ##  Character number.
        character = 94

        ##  Sequence length.
        sequence = 512

        ##  Block of character.
        block = {
            'character':nn.Sequential(nn.Conv2d(512, character, 1), nn.ReLU())
        }

        ##  Layer structure.
        layer = {
            "image convolutional":nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1]),
            "character convolutional":nn.ModuleDict({str(i):block['character'] for i in range(sequence)}),
            "recurrent":nn.GRU(character, character, 2),
            "attention":nn.TransformerEncoderLayer(d_model=character, nhead=1),
            "probable":nn.Softmax(dim=2)
        }
        self.layer = nn.ModuleDict(layer)
        pass

    def forward(self, batch):

        ##  Handle batch.
        feature = batch
        midden = {"feature":None}

        ##  Feature forward.
        midden['feature']    = self.layer['image convolutional'](feature)
        midden['feature']    = [self.layer['character convolutional'][i](midden['feature']).flatten(1,-1).unsqueeze(1) for i in self.layer['character convolutional']]
        midden['feature']    = torch.cat(midden['feature'], 1)
        midden['feature'], _ = self.layer['recurrent'](midden['feature'])
        midden['feature']    = self.layer['attention'](midden['feature'])
        midden['feature']    = self.layer['probable'](midden['feature'])

        # ##  Target forward.
        # midden['target']    = self.layer['embedded'](target)
        # midden['target']    = self.layer['probable'](midden['target'])

        ##  Handle output.
        output = midden['feature']
        return(output)

    def load(self, path):
        
        weight = torch.load(path, map_location='cpu')
        self.load_state_dict(weight)
        pass


#x = torch.randn((12,3,224,224))
#model()(x).shape

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



# import torch
# import torch.nn as nn
# x = torch.randn((2,6,3))
# y = torch.randint(0,3, (2, 6))
# x.shape
# y.shape
# xx = torch.flatten(x, 0, 1)
# yy = torch.flatten(y)
# loss = nn.CrossEntropyLoss()
# loss(xx, yy)

# y.shape
# xx.shape

##
# feature = torch.randn((16, 3, 224, 226))
# target  = torch.randint(0, 27, (16, 512))
# batch = feature, target

# test = model()
# x, y = test(batch)
# x.shape
# y.shape
