
##  Packages.
import data, network

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv")
table = data.tabulation.filter(table=table, column='mode', value='train')

##  Debug or not.
debug = True
if(debug):

    number = round(len(table)/1000)
    table  = table.sample(number)
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
loader = data.loader(train=train['dataset'], check=check['dataset'], batch=4)
if(loader.available("train") and loader.available("check")):

    print("Loader work successfully.")
    pass

##
model     = network.model()
criterion = network.criterion.cel(weight=data.process.dictionary['weight'])

##
optimizer = network.optimizer.adam(model)

##
folder   = "SOURCE/LOG"
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")

##
iteration = 30
history = {
    'train' : {"cost":[]},
    'check' : {"cost":[]}
}
for epoch in range(iteration):

    ##  Learning process.
    machine.learn(loader.train)

    if(epoch%5==0):

        machine.measure(train=loader.train, check=loader.check)
        machine.save("checkpoint")
        machine.save("measurement")

        ##  Measurement.
        measurement = machine.measurement
        
        ##  History of epoch.
        history['train']['cost'] += [measurement['train']['cost']]
        history['check']['cost'] += [measurement['check']['cost']]
        
        ##  Save the report.
        report = network.report(train=history['train'], check=history['check'])
        report.summarize()
        report.save(folder=folder)
        pass

    ##  Update checkpoint.
    machine.update('checkpoint')
    pass


# measurement['check']['likelihood'][0,:][511,:]
# measurement['check']['target'][0]
# # import numpy
# numpy.argmax(measurement['check']['likelihood'][0,:], axis=1)

