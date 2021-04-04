
##
import torch, torchvision
import torch.nn as nn

##
class model(torch.nn.Module):

    def __init__(self):

        super(model, self).__init__()
        pass

        self.layer = {
            "net"    : torchvision.models.resnet18(pretrained=True),
            "output" : nn.Linear(1000, 24)
        }
        pass

    def forward(self, feature):

        score = {}
        score['net']    = self.layer['net'](feature)
        score['output'] = self.layer['output'](score['net'])
        pass

        output = score['output']
        return(output)

    def load(self, path):
        
        weight = torch.load(path, map_location='cpu')
        self.load_state_dict(weight)
        pass
