
##  Packages.
import data

##  Every from table.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv")
table = data.tabulation.skip(table=table, column='type', value='test')

##  Create train, check.
train, check = data.validation.split(table, classification=None, ratio=0.2)
if(False):

    train['table'], check['table'] = train['table'].sample(2000), check['table'].sample(1000) 
    pass

##
train['dataset'] = data.dataset(train['table'], image=data.process.image.learn , target=data.process.target.learn )
check['dataset'] = data.dataset(check['table'], image=data.process.image.review, target=data.process.target.review)

##
loader = data.loader(train=train['dataset'], check=check['dataset'], batch=16)
loader.available("train")
loader.available("check")

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

    ##  Build model.
    machine.learn(loader.train)
    machine.measure(train=loader.train, check=loader.check)
    machine.save()
    machine.update('checkpoint')

    ##  History of epoch.
    measurement = machine.measurement
    history['train']['mae'] += [network.metric.mae(measurement['train']['target'], measurement['train']['likelihood'])]
    history['check']['mae'] += [network.metric.mae(measurement['check']['target'], measurement['check']['likelihood'])]
    
    ##  Save the report.
    report = network.report(train=history['train'], check=history['check'], folder=folder)
    report.summarize()
    report.save()
    pass


