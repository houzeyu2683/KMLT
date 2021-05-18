
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
loader = data.loader(test=test['dataset'], batch=8)

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



