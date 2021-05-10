

##
##
import torch, torchvision
import torch.nn as nn


##
##  Class for model, case by case.
class model(torch.nn.Module):

    def __init__(self):

        super(model, self).__init__()
        pass
        
        ##  Character number.
        self.character = 94

        ##  Sequence length.
        self.sequence = 512

        ##  Embedding dimension.
        self.embedding = 32

        ##  Layer structure.
        layer = {
            "convolutional":nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1], nn.Sigmoid()),
            "embedded":nn.Embedding(self.character, self.embedding)
        }
        self.layer = nn.ModuleDict(layer)
        pass

    def forward(self, batch):

        ##  Handle batch.
        feature, target = batch

        ##  Forward.
        record = {}
        record['convolutional feature'] = self.layer['convolutional'](feature).flatten(1,-1) * self.character
        record['convolutional code'] = torch.as_tensor(record['convolutional feature'], dtype=torch.long)
        record['embedded code'] = self.layer['embedded'](record['convolutional code'])
        record['embedded target']  = self.layer['embedded'](target)
        
        
        ##  Handle output.
        output = record['embedded code'], record['embedded target'], record['convolutional code']
        return(output)

    def load(self, path):
        
        weight = torch.load(path, map_location='cpu')
        self.load_state_dict(weight)
        pass


# ##
# ##
# import torch, torchvision
# import torch.nn as nn
# number = {
#     "batch":12,
#     "sequence":512,
#     "embedding":4,
#     "character":26
# }
# batch = torch.randn((number['batch'], 3, 224, 224)), torch.randint(1, number['character'], (number['batch'], number['sequence']))
# image, target = batch



# layer_01 = nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1], nn.Sigmoid())
# layer_02 = nn.Embedding(number['character'], number['embedding'])


# code = torch.as_tensor(layer_01(image).flatten(1,-1) * number['character'], dtype=torch.long)
# # #code = midden.clone()
# # #code = code.detach()
# # code = code.flatten(1,-1) * number['character']
# # code = torch.as_tensor(code, dtype=torch.long)

# code_image.shape
# code_image = layer_02(code)
# code_text  = layer_02(target)

# loss = nn.MSELoss()
# output = loss(code_image, code_text)


# image.shape




# torch.randint(1, number['character'], 15)

# torch.randint(0,94, (12, 512)).
# batch = torch.randn((12, 3, 224, 224)), 
# feature = batch
# x = feature
# layer_image = nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1])

# x = layer_image(x)
# x = nn.Sigmoid()(x)
# y = torch.round(x*94)
# y = y.detach()
# y = torch.tensor(y, dtype=torch.long)
# y

# m = model.layer['embedded']
# m.weight[0,:]
# m(torch.tensor(0))


# # number = {
# #     "character":94,
# #     "sequence":512
# # }
# model = nn.Conv2d(1, 12 , 1, (1,1))
# input1 = torch.randn(12, 1, 100, 128)
# x = model(input1)
# x = x[0,:]
# input2 = torch.randn(12, 100, 128)
# cos = nn.CosineSimilarity(dim=2, eps=1e-6)
# output = cos(x, input2)
# output.shape
# output.bac

# # block = nn.ModuleDict({
# #     "01":nn.Sequential(
# #         nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
# #         nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
# #         nn.ReLU(inplace=True),
# #         nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
# #         nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# #     ),
# #     "02":nn.Sequential(
# #         nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
# #         nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
# #         nn.ReLU(inplace=True),
# #         nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
# #         nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# #     ),
# #     "03":nn.Sequential(
# #         nn.Conv2d(128, 94, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
# #         nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
# #         nn.ReLU(inplace=True),
# #         nn.Conv2d(94, 94, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
# #         nn.BatchNorm2d(94, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# #     ),
# #     "04":nn.AdaptiveAvgPool2d(output_size=(1,1))
# # })

# # layer = {
# #     'image':nn.ModuleDict({str(i):nn.Sequential(block['01'],block['02'],block['03'],block['04']) for i in range(512)})
# # }

# # midden = [layer['image'][i](feature) for i in layer['image']]


# # layer['image']["0"](feature)

# # x = block['01'](x)
# # x = block['02'](x)
# # x = block['03'](x)
# # x = block['04'](x)

