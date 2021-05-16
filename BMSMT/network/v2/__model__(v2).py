

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


import torch, torchvision
import torch.nn as nn
batch    = torch.randn((12,3,224,224))
feature  = batch

##  Character number.
character = 94

##  Sequence length.
sequence = 512

block = {}
nn.ModuleDict({
    "01":nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ),
    "02":nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ),
    "03":nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ),
    "04":nn.AdaptiveAvgPool2d(output_size=(1,1))    
})

ll = nn.Sequential(

    ##
    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

    ##
    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),    

    ##
    nn.Conv2d(128, 94, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
    nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(inplace=True),
    nn.Conv2d(94, 94, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
    nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 

    ##
    nn.AdaptiveAvgPool2d(output_size=(1,1))
)
ll(feature).shape

# ##  Layer
# layer = {
#     "image convolutional": nn.Sequential(
#         *list(torchvision.models.resnet18(True).children())[:-1], 
#         nn.Conv2d(512, character, kernel_size=(1, 1), stride=(1, 1), padding=(0,0), bias=False)
#     )
# }

# bb = layer['image convolutional'](feature)
# bb.shape



import torchvision, torch
import torch.nn as nn

batch    = torch.randn((12,3,224,224))
feature  = batch

##  Character number.
character = 94

##  Sequence length.
sequence = 512


character_layer = nn.ModuleDict({
    "01":nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ),
    "02":nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ),
    "03":nn.Sequential(
        nn.Conv2d(128, 94, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(94, 94, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ),
    "04":nn.AdaptiveAvgPool2d(output_size=(1,1))    
})

x = character_layer['01'](feature)
x = character_layer['02'](x)
x = character_layer['03'](x)
x = character_layer['04'](x)

x.flatten(1,-1)





encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 1)
out = transformer_encoder(src)

out.shape