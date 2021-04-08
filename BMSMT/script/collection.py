
##
import pandas, os

##
path = {
    'train':{
        'csv':{
            'label':"../DATA/BMSMT/TRAIN/CSV/LABEL.csv"
        }
    },
    'test':{
        'csv':{
            'label':"../DATA/BMSMT/TEST/CSV/LABEL.csv"
        }
    }
}

##
table = {
    'train':{
        "label":pandas.read_csv(path['train']['csv']['label'])
    },
    "test":{
        'label':pandas.read_csv(path['test']['csv']['label'])    
    }
}
table['train']['label']['type'] ='train'
table['test']['label']['type']  ='test'
table['label'] = pandas.concat([table['train']['label'], table['test']['label']])
table['label'].head()

##
folder = 'SOURCE/CSV/'
title  = "LABEL.csv"
path   = os.path.join(folder, title)
os.makedirs(folder, exist_ok=True)
table['label'].to_csv(path, index=False)

