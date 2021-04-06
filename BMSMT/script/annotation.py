
##  Packages.
import os, pandas, re, numpy, tqdm, pickle

##
train = {
    "table":{
        'label':pandas.read_csv("../DATA/BMSMT/TRAIN/CSV/LABEL.csv"),
        'count':pandas.read_csv('SOURCE/TRAIN/CSV/COUNT.csv'),
        'order':pandas.read_csv('SOURCE/TRAIN/CSV/ORDER.csv')
    }
}
test = {
    'table':{
        'label':pandas.read_csv("../DATA/BMSMT/TEST/CSV/LABEL.csv")
    }
}

##
train['table']['label']['image'] = ["../DATA/BMSMT/TRAIN/PNG/" + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png' for i in train['table']['label']['image_id']]
test['table']['label']['image']  = ["../DATA/BMSMT/TEST/PNG/" + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png' for i in test['table']['label']['image_id']]
train['table']['label']['type'] = 'train'
test['table']['label']['type']  = 'test'

##
train['table']['annotation'] = pandas.concat([train['table']['label'], train['table']['count'], train['table']['order']], axis=1)

##
annotation = pandas.concat([train['table']['annotation'], test['table']['label']],axis=0).reset_index(drop=True)
annotation = annotation.fillna(0)

##
source = 'SOURCE/CSV/'
os.makedirs(source, exist_ok=True)
annotation.to_csv(os.path.join(source, 'ANNOTATION.csv'), index=False)

