
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

##  Size.
train['size'] = len(train['table'])
check['size'] = len(check['table'])

##  Initialize the dataset.
train['dataset'] = data.dataset(train['table'], image=data.process.image.learn , target=data.process.target.learn )
check['dataset'] = data.dataset(check['table'], image=data.process.image.review, target=data.process.target.review)


##
loader = data.loader(train=train['dataset'], check=check['dataset'], batch=1)
if(loader.available("train") and loader.available("check")):

    print("Loader work successfully.")
    pass

##
import network

##
model     = network.model()
criterion = network.criterion.cel(weight=[i for i in reversed(range(94))])

##
optimizer = network.optimizer.adam(model)

##
folder   = "SOURCE/LOG"
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")

##
iteration = 30
history = {
    'train' : {"cel":[]},
    'check' : {"cel":[]}
}
for epoch in range(iteration):

    ##  Learning process.
    machine.learn(loader.train)
    machine.measure(train=loader.train, check=loader.check)
    machine.save("checkpoint")
    machine.save("measurement")
    machine.update('checkpoint')

    ##  Measurement.
    measurement = machine.measurement
    
    ##  History of epoch.
    item = [network.metric.cel(measurement['train']['target'][i,:], measurement['train']['likelihood'][i,:,:], label=range(94)) for i in range(train['size'])]
    history['train']['cel'] += [sum(item)/len(item)]
    item = [network.metric.cel(measurement['check']['target'][i,:], measurement['check']['likelihood'][i,:,:], label=range(94)) for i in range(check['size'])]
    history['check']['cel'] += [sum(item)/len(item)]
    
    ##  Save the report.
    report = network.report(train=history['train'], check=history['check'])
    report.summarize()
    report.save(folder=folder)
    pass



measurement['check']['likelihood'][0,:][511,:]
measurement['check']['target'][0]
# import numpy
numpy.argmax(measurement['check']['likelihood'][0,:], axis=1)

