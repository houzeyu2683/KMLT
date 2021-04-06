
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
source = 'SOURCE/TRAIN/CSV/'
os.makedirs(source, exist_ok=True)
count.to_csv(os.path.join(source, "COUNT.csv"), index=False)

