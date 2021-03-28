import os, pandas, re
train = {
    "csv"  : {},
    "list" : {}
}
train['csv']['label']  = pandas.read_csv("DATA/BMSMT/TRAIN/CSV/LABEL.csv")
train['list']['label'] = [i.split("InChI=1S/")[1] for i in train['csv']['label']['InChI']]
train['list']['label'] = [i.split('/')[0] for i in train['list']['label']]
train['list']['label'] = [re.sub("[0-9]", "", i) for i in train['list']['label']]
train['list']['length'] = []

i = 'CHClNOS'
i

w = i
for i, j in 

