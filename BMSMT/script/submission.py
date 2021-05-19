
##  Packages.
import data, network

##  Load table and skip real test data.
table = data.tabulation.read("SOURCE/CSV/ANNOTATION.csv")
table = data.tabulation.filter(table=table, column='mode', value='test')

##  Debug or not.
debug = True
if(debug):

    number = round(len(table)/4000)
    table  = table.sample(number)
    pass

##  Group.
group = {}
group['size'] = 4
group['table']   = [value['check'] for _, value in data.validation.fold(table=table, size=group['size']).items()]
group['dataset'] = [data.dataset(i, image=data.process.image.review , text=data.process.text.tokenize) for i in group['table']] 
group['loader']  = [data.loader(test=i, batch=8).test for i in group['dataset']]

##
model = network.model()

##
machine = network.machine(model=model, device='cuda')
machine.load(what='weight', path="SOURCE/LOG/2.checkpoint")

##
# prediction = machine.predict(loader.test, length=128)

##
def submit(index, machine, group):

    prediction = machine.predict(group['loader'][index], length=128)
    group['table'][index]['InChI'] = prediction
    submission = group['table'][index][['image_id', 'InChI']]
    submission.to_csv("SOURCE/SUBMISSION-" + str(index) + ".csv", index=False)
    print("Finish the {} thread.\n".format(index))
    return



# ##
# test['table']['InChI'] = prediction

# ##
# submission = test['table'][['image_id', 'InChI']]
# submission.to_csv("SOURCE/SUBMISSION.csv", index=False)


import threading, time
thread = []
for index in range(group['size']):
    
    thread += [threading.Thread(target=submit, args=(index, machine, group))]
    thread[index].start()
    time.sleep(5)
    pass



