import os, pandas, re, numpy
train = {
    "csv"  : {},
    "list" : {}
}
train['csv']['label']  = pandas.read_csv("DATA/BMSMT/TRAIN/CSV/LABEL.csv")

train['csv']['label_title']  = "InChI=1S/"
train['csv']['label_*/text/*'] = [i.split("InChI=1S/")[1] for i in train['csv']['label']['InChI']]
train['list']['element'] = [i.split('/')[0] for i in train['csv']['label_*/text/*']]

i = train['list']['element'][0]
#character = 'CHClNOS'

def element(character):
    word = []
    for index, letter in enumerate(character):
        if( re.compile("[A-Z]").search(letter) ):
            word += [letter]
        if( re.compile("[a-z]").search(letter) ):
            word[-1] = word[-1] + letter
    return(word)

