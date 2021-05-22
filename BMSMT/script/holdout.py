
##  Packages.
import data, network

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv", number=5000)
table = data.tabulation.filter(table=table, column='mode', value='train')

##  Split table to train and check type.
train, check = data.validation.split(table, classification=None, ratio=0.1)

##  Initialize the dataset.
train['dataset'] = data.dataset(train['table'], image=data.process.image.learn , text=data.process.text.learn)
check['dataset'] = data.dataset(check['table'], image=data.process.image.review, text=data.process.text.review)

##
loader = data.loader(train=train['dataset'], check=check['dataset'], batch=8)
if(loader.available("train") and loader.available("check")):
    
    print("Loader work successfully.")
    pass

##
vocabulary = data.process.vocabulary.load("SOURCE/PICKLE/VOCABULARY.pickle")

##
model = network.model(vocabulary=vocabulary)
# model(next(iter(loader.train)))

criterion = network.criterion.cel(ignore=vocabulary['<pad>'])

##
optimizer = network.optimizer.adam(model)

##
folder   = "SOURCE/LOG"

##
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")

##
iteration = 1
history = {
    'train' : {"cost":[]},
    'check' : {"cost":[]}
}
for epoch in range(iteration):

    ##  Learning process.
    machine.learn(loader.train)

    if(epoch%1==0):

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

    ##  Update.
    machine.update('schedule')
    machine.update('checkpoint')
    pass



