
##  Packages.
import data

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv")
table = data.tabulation.skip(table=table, column='mode', value='test')

##  Debug or not.
debug = True
if(debug):

    sample = round(len(table)/15000)
    table  = table.sample(sample)
    pass

##  Split table to train and check type.
train, check = data.validation.split(table, classification=None, ratio=0.2)

##  Initialize the dataset.
train['dataset'] = data.dataset(train['table'], image=data.process.image.learn , target=data.process.target.learn )
check['dataset'] = data.dataset(check['table'], image=data.process.image.review, target=data.process.target.review)


##
loader = data.loader(train=train['dataset'], check=check['dataset'], batch=4)
if(loader.available("train") and loader.available("check")):

    print("Loader work successfully.")
    pass

##
import network

##
model     = network.model()
criterion = network.criterion.mae()

##
optimizer = network.optimizer.sgd(model)

##
folder   = "SOURCE/LOG"
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")

##
iteration = 2
history = {
    'train' : {"mae":[]},
    'check' : {"mae":[]}
}
for epoch in range(iteration):
    break

    ##  Learning process.
    machine.learn(loader.train)
    machine.measure(train=loader.train, check=loader.check)
    machine.save("checkpoint")
    machine.save("measurement")
    machine.update('checkpoint')

    ##  History of epoch.
    measurement = machine.measurement
    history['train']['mae'] += [network.metric.mae(measurement['train']['target'].flatten(), measurement['train']['likelihood'].flatten())]
    history['check']['mae'] += [network.metric.mae(measurement['check']['target'].flatten(), measurement['check']['likelihood'].flatten())]
    
    ##  Save the report.
    report = network.report(train=history['train'], check=history['check'])
    report.summarize()
    report.save()
    pass

report.summary
import torch
import torch.nn as nn

measurement['train']['target']

torch.nn.KLDivLoss(measurement['train']['target'], measurement['train']['likelihood'][0,:,:])
torch.squeeze(measurement['train']['target']).size()

network.metric.mae(
    measurement['check']['target'].flatten(),
    measurement['check']['likelihood'].flatten()
)

x = torch.nn.KLDivLoss()
x(torch.tensor(measurement['train']['target']), torch.tensor(measurement['train']['likelihood']))


x = torch.tensor(measurement['check']['likelihood'][0,:,:])
x.shape

torch.argmax(x, dim=2)



