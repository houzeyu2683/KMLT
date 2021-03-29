import os, pandas, re, numpy, tqdm

def capture(text):
    element = []
    for index, letter in enumerate(text):
        if( re.compile("[A-Z]").search(letter) ):
            element += [letter]
        if( re.compile("[a-z]").search(letter) ):
            element[-1] = element[-1] + letter
    return(element)

data    = pandas.read_csv("DATA/BMSMT/TRAIN/CSV/LABEL.csv")
element = []
for index, row in tqdm.tqdm(data.iterrows(), total=len(data)):
    text = row['InChI'].split("InChI=1S/")[1].split('/')[0]
    element += capture(text)

[i for i in element]
x = element[0]

# train = {
#     "csv"  : {},
#     "list" : {}
# }
# train['csv']['label']  = pandas.read_csv("DATA/BMSMT/TRAIN/CSV/LABEL.csv")

# train['csv']['label_title']  = "InChI=1S/"
# train['csv']['label_*/text/*'] = [i.split("InChI=1S/")[1] for i in train['csv']['label']['InChI']]
# train['list']['element'] = [i.split('/')[0] for i in train['csv']['label_*/text/*']]

# i = train['list']['element'][0]
# #character = 'CHClNOS'

