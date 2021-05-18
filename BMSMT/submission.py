
##
import sys
sys.path.append("../")

##  Packages.
import data, network

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv")
table = data.tabulation.filter(table=table, column='mode', value='test')

##  Debug or not.
debug = False
if(debug):

    number = round(len(table)/4000)
    table  = table.sample(number).reset_index(drop=True)
    pass

##  Split table to train and check type.
test = data.validation.split(table, classification=None, ratio=None)

##  Initialize the dataset.
test['dataset'] = data.dataset(test['table'], image=data.process.image.review , text=data.process.text.tokenize)

##
loader = data.loader(test=test['dataset'], batch=4)

##
model = network.model()

##
machine = network.machine(model=model, device='cuda')
machine.load(what='weight', path="SOURCE/LOG/4.checkpoint")

##
prediction = machine.predict(loader.test, length=128)
test['table']['InChI'] = prediction


submission = test['table'][['image_id', 'InChI']]
submission.to_csv("SOURCE/SUBMISSION.csv", index=False)




# import torch
# x = torch.randn((256, 4, 512))
# x[:, 4:5, :].shape
# model.eval()
# model.load_state_dict(torch.load("SOURCE/LOG/2.checkpoint"))

# prediction = []
# for batch in tqdm.tqdm(loader.test):

#     image, _ = batch
#     image = image.to('cuda')
#     prediction += [model.convert(image)]
#     pass

# test['table']['InChI'] = prediction
# test['table'][['image_id', 'InChI']].to_csv("SOURCE/sub.csv", index=False)


# batch = next(iter(loader.train))
# batch = batch[0][0:1, :,:].to('cuda'), batch[1]
# image = batch[0][0:1, :,:].to('cuda')
# x = machine.model.to('cuda').convert(image)
# x = model
# len(x)

# x.split(',')
# batch[0][0:1, :,:].shape
##
# iteration = 200
# history = {
#     'train' : {"cost":[]},
#     'check' : {"cost":[]}
# }
# for epoch in range(iteration):

#     ##  Learning process.
#     machine.learn(loader.train)

#     if(epoch%1==0):

#         machine.measure(train=loader.train, check=loader.check)
#         machine.save("checkpoint")
#         machine.save("measurement")

#         ##  Measurement.
#         measurement = machine.measurement
        
#         ##  History of epoch.
#         history['train']['cost'] += [measurement['train']['cost']]
#         history['check']['cost'] += [measurement['check']['cost']]
        
#         ##  Save the report.
#         report = network.report(train=history['train'], check=history['check'])
#         report.summarize()
#         report.save(folder=folder)
#         pass

#     ##  Update.
#     machine.update('schedule')
#     machine.update('checkpoint')
#     pass
