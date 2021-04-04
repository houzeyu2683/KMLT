
##  Packages.
import os, pandas, re, numpy, tqdm, pickle

##
train = {
    "csv":{
        'label':pandas.read_csv("../DATA/BMSMT/TRAIN/CSV/LABEL.csv"),
        'count':pandas.read_csv('SOURCE/TRAIN/CSV/COUNT.csv'),
        'order':pandas.read_csv('SOURCE/TRAIN/CSV/ORDER.csv')
    }
}

##
annotation          = pandas.concat([train['csv']['label'], train['csv']['count'], train['csv']['order']], axis=1)
annotation['image'] = ["../DATA/BMSMT/TRAIN/PNG/" + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png' for i in annotation['image_id']]
annotation.head()
source = 'SOURCE/TRAIN/CSV/'
os.makedirs(source, exist_ok=True)
annotation.to_csv(os.path.join(source, 'ANNOTATION.csv'), index=False)

