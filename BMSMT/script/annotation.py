
##
import pandas, os, tqdm

##
group = []
for mode in ['train', 'test']:

    if(mode=='train'):

        ##
        table = pandas.read_csv("../DATA/BMSMT/TRAIN/CSV/LABEL.csv")
        table['mode'] = 'train'
        
        ##
        folder = "../DATA/BMSMT/TRAIN/PNG/"
        table['image']  = [folder + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png' for i in table['image_id']]
        table['length'] = [len(i) for i in table['InChI']]
        group += [table]
        pass

    if(mode=='test'):

        ##
        table = pandas.read_csv("../DATA/BMSMT/TEST/CSV/LABEL.csv")
        table['mode']  = 'test'
        table['InChI'] = "InChI=1S/"        

        ##
        folder = "../DATA/BMSMT/TEST/PNG/"
        table['image']  = [folder + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png' for i in table['image_id']]
        table['length'] = [len(i) for i in table['InChI']]        
        group += [table]
        pass

##
annotation = pandas.concat(group)
annotation.tail()

##
annotation['InChI'] = [i+'.' for i in annotation['InChI']]


##
path = "SOURCE/CSV/ANNOTATION.csv"
os.makedirs(os.path.dirname(path), exist_ok=True)
annotation.to_csv(path, index=False)

