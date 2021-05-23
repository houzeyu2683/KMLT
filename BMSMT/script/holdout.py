
##  Packages.
import data

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

    batch = next(iter(loader.train))    
    print("Loader work successfully.")
    pass

##
vocabulary = data.vocabulary.load("SOURCE/PICKLE/VOCABULARY.pickle")

##
import network

##
model = network.model(vocabulary=vocabulary)
# model.convert(batch[0])

criterion = network.criterion.cel(ignore=vocabulary['<pad>'])

##
optimizer = network.optimizer.adam(model)

##
folder   = "SOURCE/LOG"

##
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")
machine.load(what='weight', path='SOURCE/LOG/0.checkpoint')

##
iteration = 20
history = {
    'check' : {"cost":[], "score":[]}
}
for epoch in range(iteration):

    ##  Learning process.
    machine.learn(loader.train)

    if(epoch%1==0):

        machine.measure(check=loader.check)
        machine.save("checkpoint")
        machine.save("measurement")

        ##  Measurement.
        measurement = machine.measurement
        
        ##  History of epoch.
        # history['train']['cost'] += [measurement['train']['cost']]
        history['check']['cost'] += [measurement['check']['cost']]
        history['check']['score'] += [measurement['check']['score']]
        
        ##  Save the report.
        report = network.report(check=history['check'])
        report.summarize()
        report.save(folder=folder)
        pass

    ##  Update.
    machine.update('schedule')
    machine.update('checkpoint')
    pass



