
##
import data, network

##
train = {}
check = {}

##
train['table'], check['table'] = data.validation.split(data.table.read("SOURCE/TRAIN/CSV/ANNOTATION.csv"))

##
train['dataset'] = data.dataset(train['table'], image=data.process.image.learn, target=data.process.target.learn)
check['dataset'] = data.dataset(check['table'], image=data.process.image.review, target=data.process.target.review)
if(True):
    train['dataset'].__getitem__(1)
    check['dataset'].__getitem__(1)
    pass

##
loader = data.loader(train=train['dataset'], check=check['dataset'])
if(True):
    next(iter(loader.train))
    next(iter(loader.check))
    pass

model     = network.model
help(network.model)
x = next(iter(loader.check))
len(x)
model(x['image'])
criterion = network.criterion.entropy()
optimizer = network.optimizer.adam(model)
