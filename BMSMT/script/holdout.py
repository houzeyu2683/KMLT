
##  Packages.
import data

##  Every from table.
table = data.table.read("SOURCE/CSV/ANNOTATION.csv")

##  Create train, check, test.
train, check, test = data.validation.split(table)
if(False):

    train['table'], check['table'], test['table'] = train['table'].sample(1000), check['table'].sample(1000), test['table'].sample(1000) 
    pass

##
train['dataset'] = data.dataset(train['table'], image=data.process.image.learn , target=data.process.target.learn )
check['dataset'] = data.dataset(check['table'], image=data.process.image.review, target=data.process.target.review)
test['dataset']  = data.dataset(test['table'] , image=data.process.image.review, target=data.process.target.review)

##
loader = data.loader(train=train['dataset'], check=check['dataset'], batch=256)
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
folder = "SOURCE/LOG"
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")

##
iteration = 10
history = {
    'train' : {"mae":[]},
    'check' : {"mae":[]}
}
for epoch in range(iteration):

    ##
    machine.learn(loader.train)
    machine.measure(train=loader.train, check=loader.check)
    machine.update('checkpoint')

    ##
    measurement = machine.measurement
    history['train']['mae'] += [network.metric.mae(measurement['train']['target'], measurement['train']['likelihood'])]
    history['check']['mae'] += [network.metric.mae(measurement['check']['target'], measurement['check']['likelihood'])]
    
    ##
    report = network.report(train=history['train'], check=history['check'])
    report.summarize()
    report.save(folder=folder)
    pass

