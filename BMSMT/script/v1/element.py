
##  Packages.
import os, pandas, re, numpy, tqdm, pickle

##  Capture function.
def capture(label):

    ##  Initial object.
    element = {}
    end     = len(label)-1

    ##  For each letter in label.
    for index, letter in enumerate(label):

        ##  Frist letter case.
        if index==0:

            item = {"name":letter, "count":""}
            continue

        ##  Middle letter case.
        if index!=end:

            if re.compile('[A-Z]').search(letter):

                element[str(index)] = item
                item = {"name":letter, "count":""}
                pass

            if re.compile('[a-z]').search(letter):

                item['name'] += letter
                pass

            if re.compile('[0-9]').search(letter):

                item['count'] += letter
                pass
        
        ##  Last letter case.
        if index==end:

            if re.compile('[A-Z]').search(letter):

                element[str(index)] = item
                item = {"name":letter, "count":""}
                element[str(index+1)] = item
                pass

            if re.compile('[a-z]').search(letter):

                item['name'] += letter
                element[str(index)] = item                
                pass

            if re.compile('[0-9]').search(letter):

                item['count'] += letter
                element[str(index)] = item
                pass

    name  = []
    count = []
    for _, value in element.items():

        name.append(value['name'])
        pass

        if value['count']=='':

            count.append(1)
            pass
    
        else:
    
            count.append(int(value['count']))
            pass
    
    output = dict(zip(name, count))
    return(output)

##  
table = {
    "label":pandas.read_csv("SOURCE/CSV/LABEL.csv")
}

##
group = {}
for index, item in tqdm.tqdm(table['label'].iterrows(), total=len(table['label']), leave=False):

    origin = item['InChI'].split("InChI=1S/")[1].split('/')[0]
    group.update({index : capture(origin)})
    pass

group = [group[key] for key in tqdm.tqdm(group, leave=False)]

##
name = []
for i in tqdm.tqdm(group, leave=False):

    name = list(set(name + list(i.keys())))
    pass

##  Element.
element = {"name":name, "group":group}
folder  = "SOURCE/PICKLE/"
title   = "ELEMENT.pickle"
path    = os.path.join(folder, title)
os.makedirs(folder, exist_ok=True)
with open(path, 'wb') as paper:

    pickle.dump(element, paper)
    pass

