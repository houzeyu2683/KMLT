

##
##  Packages.
import torch, torchvision
import torch.nn as nn


##
##  The [model] class.
class model(torch.nn.Module):

    def __init__(self):

        super(model, self).__init__()
        pass

        self.layer = nn.Sequential(
            torchvision.models.resnet18(pretrained=True),
            nn.Linear(1000, 24) 
        )
        pass

    def forward(self, feature):

        score = self.layer(feature)
        pass

        output = score
        return(output)

    def load(self, path):
        
        weight = torch.load(path, map_location='cpu')
        self.load_state_dict(weight)
        pass

