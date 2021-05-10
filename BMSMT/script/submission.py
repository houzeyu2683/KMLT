
##  Packages.
import data, network

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv")
table = data.tabulation.filter(table=table, column='mode', value='test')

##  Split table to train and check type.
test = data.validation.split(table, classification=None, ratio=None)

##
test['dataset'] = data.dataset(test['table'], image=data.process.image.review, target=data.process.target.review)

##
loader = data.loader(test=test['dataset'], batch=16)
if(loader.available("test")):

    print("Loader work successfully.")
    pass

##
weight = "./SOURCE/LOG/6.checkpoint"
model  = network.model()
model.load(weight)

##
criterion = network.criterion.mae()

##
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=None, checkpoint="0")


model.layer['embedded'](torch.tensor(range(94)))


import torch

embeddings = torch.nn.Embedding(20, 20)
my_sample = torch.randint(0, 20, (12, 8))
vv = embeddings(my_sample)
torch.argmax()
vv.argmax(dim=2)
distance = torch.norm(embeddings.weight.data - my_sample, dim=1)
nearest = torch.argmin(distance)
model

import torch.nn as nn
input1 = torch.randn(1, 100, 128)
input2 = torch.randn(1, 100, 128)
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
output = cos(input1, input2)
output.shape


