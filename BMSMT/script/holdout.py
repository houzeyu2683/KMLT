
##  Packages.
import data, network

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv")
table = data.tabulation.filter(table=table, column='mode', value='train')

##  Debug or not.
debug = True
if(debug):

    number = round(len(table)/4000)
    table  = table.sample(number).reset_index(drop=True)
    pass

##  Split table to train and check type.
train, check = data.validation.split(table, classification=None, ratio=0.2)

##  Initialize the dataset.
train['dataset'] = data.dataset(train['table'], image=data.process.image.learn , text=data.process.text.tokenize)
check['dataset'] = data.dataset(check['table'], image=data.process.image.review, text=data.process.text.tokenize)

##
loader = data.loader(train=train['dataset'], check=check['dataset'], batch=8)
if(loader.available("train") and loader.available("check")):
    
    batch = next(iter(loader.train))
    print("Loader work successfully.")
    pass

##
model     = network.model()
criterion = network.criterion.cel()

##
optimizer = network.optimizer.sgd(model)

##
folder   = "SOURCE/LOG"
machine  = network.machine(model=model, optimizer=optimizer, criterion=criterion, device='cuda', folder=folder, checkpoint="0")

# batch = next(iter(loader.train))
# batch = batch[0][0:1, :,:].to('cuda'), batch[1]
# image = batch[0][0:1, :,:].to('cuda')
# x = machine.model.to('cuda').convert(image)
# x = model
# len(x)

# x.split(',')
# batch[0][0:1, :,:].shape
##
iteration = 200
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


#measurement['check'][''][0,:][511,:]
# measurement['check']['target'][0]
# # import numpy
# numpy.argmax(measurement['check']['likelihood'][0,:], axis=1)

