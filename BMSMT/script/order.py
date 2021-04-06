
##  Packages.
import os, pandas, re, numpy, tqdm, pickle

##
train = {
    'pickle':{
        'element':'SOURCE/TRAIN/PICKLE/ELEMENT.pickle'
    }
}

##
with open(train['pickle']['element'], 'rb') as paper:

    name, group = pickle.load(paper) 
    pass

##  
order = dict(zip(name, [[] for i in name]))
for item in group:

    for key in order:

        order[key] += [0]
        pass

    for index, key in enumerate(item, 1):
    
        order[key][-1] = index
        pass

order = pandas.DataFrame(order).reset_index(drop=True)
order.columns = ["$"+i for i in order.columns]

##
source = "SOURCE/TRAIN/CSV"
os.makedirs(source, exist_ok=True)
order.to_csv(os.path.join(source, "ORDER.csv"), index=False)

