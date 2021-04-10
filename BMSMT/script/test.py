
##
import data, network

##
test = {}
test['table']   = data.tabulation.skip(table=data.tabulation.read("SOURCE/CSV/ANNOTATION.csv"), column='type', value='train')
test['dataset'] = data.dataset(test['table'], image=data.process.image.review, target=data.process.target.review)

##
loader = data.loader(test=test['dataset'], batch=8)

##
model  = network.model()
model.load(path='SOURCE/LOG/3.weight')

##
machine = network.machine(model=model, device='cuda', folder="SOURCE/PICKLE", checkpoint='test')
machine.measure(test=loader.test)
machine.save()

