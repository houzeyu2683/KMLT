

##
##  Packages.
import PIL.Image, os, torch, numpy
from torchvision import transforms as kit


##
##  Class for process image, case by case.
class image:

    def learn(item):

        mu       = (0.5,0.5,0.5)
        sigma    = (0.5,0.5,0.5)
        size     = (224, 224)
        pipeline = [
            kit.RandomRotation((0, 360)),
            kit.RandomHorizontalFlip(0.5),
            kit.RandomVerticalFlip(0.5),
            kit.Resize(size),
            kit.ToTensor(),
            kit.Normalize(mean = mu, std = sigma)
        ]
        action = kit.Compose(pipeline)
        output = action(item).type(torch.float)
        return(output)

    def review(item):

        mu       = (0.5, 0.5, 0.5)
        sigma    = (0.5, 0.5, 0.5)
        size     = (224, 224)
        pipeline = [
            kit.Resize(size),
            kit.ToTensor(),
            kit.Normalize(mean = mu, std = sigma)      
        ]
        action = kit.Compose(pipeline)
        output = action(item).type(torch.float)
        return(output)
