
##  Packages.
import os, pandas, re, numpy, tqdm, pickle

##
with open('SOURCE/PICKLE/ELEMENT.pickle', 'rb') as paper:

    element = pickle.load(paper) 
    pass

##
name  = element['name']
group = element['group']

##  
order = dict(zip(name, [[] for i in name]))
for item in group:

    for key in order:

        order[key] += [-1]
        pass

    for index, key in enumerate(item, 1):
    
        order[key][-1] = index
        pass

##
order = pandas.DataFrame(order).reset_index(drop=True)
order.columns = ["$"+i for i in order.columns]

##
folder = "SOURCE/CSV"
title  = "ORDER.csv"
path   = os.path.join(folder, title)
os.makedirs(folder, exist_ok=True)
order.to_csv(path, index=False)

