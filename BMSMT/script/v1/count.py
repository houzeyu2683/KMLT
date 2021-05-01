
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
count = {i:[] for i in name}
for key in count:

    for item in tqdm.tqdm(group, leave=False):

        try:

            count[key] += [item[key]]
            pass

        except:

            count[key] += [0]
            pass

count = pandas.DataFrame(count).reset_index(drop=True)
count.columns = ["#"+i for i in count.columns]
count.head()

##
folder = 'SOURCE/CSV/'
title  = 'COUNT.csv'
path   = os.path.join(folder, title)
os.makedirs(folder, exist_ok=True)
count.to_csv(path, index=False)

