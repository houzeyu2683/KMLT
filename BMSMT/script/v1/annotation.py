
##  Packages.
import os, pandas, re, numpy, tqdm, pickle

##
table = {
    'label':pandas.read_csv("SOURCE/CSV/LABEL.csv"),
    'count':pandas.read_csv('SOURCE/CSV/COUNT.csv'),
    'order':pandas.read_csv('SOURCE/CSV/ORDER.csv')
}

image = []
for index, item in tqdm.tqdm(table["label"].iterrows(), total=len(table["label"]), leave=False):

    i = item['image_id']
    pass

    if(item['type']=='train'):
        
        image += ["../DATA/BMSMT/TRAIN/PNG/" + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png']
        pass
    
    if(item['type']=='test'):

        image += ["../DATA/BMSMT/TEST/PNG/" + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png']
        pass

table["label"]['image'] = image

##
table['annotation'] = pandas.concat([table['label'], table['count'], table['order']], axis=1).reset_index(drop=True)

##
annotation = table['annotation']
folder = 'SOURCE/CSV/'
title  = 'ANNOTATION.csv'
path   = os.path.join(folder, title)
os.makedirs(folder, exist_ok=True)
annotation.to_csv(path, index=False)

