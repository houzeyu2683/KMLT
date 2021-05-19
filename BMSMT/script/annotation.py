
##  Packages.
import pandas, os, tqdm

##  Group of table of data.
group = []
for mode in ['train', 'test']:

    if(mode=='train'):

        ##  Load table.
        table = pandas.read_csv("../DATA/BMSMT/TRAIN/CSV/LABEL.csv")
        table['mode'] = 'train'
        
        ##  Information.
        folder = "../DATA/BMSMT/TRAIN/PNG/"
        table['image']  = [folder + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png' for i in table['image_id']]
        table['length'] = [len(i) for i in table['InChI']]

        ##  Append to group.
        group += [table]
        pass

    if(mode=='test'):

        ##  Load table.
        table = pandas.read_csv("../DATA/BMSMT/TEST/CSV/LABEL.csv")
        table['mode']  = 'test'
        table['InChI'] = "InChI=1S/"        

        ##  Information.
        folder = "../DATA/BMSMT/TEST/PNG/"
        table['image']  = [folder + i[0] + '/' + i[1] + '/' + i[2] + '/' + i + '.png' for i in table['image_id']]
        table['length'] = [len(i) for i in table['InChI']]        

        ##  Append to group.
        group += [table]
        pass

##  Combination.
annotation = pandas.concat(group).reset_index(drop=True)

##  Label.
label = []
for item in annotation['InChI']:

    item = item.split("InChI=1S/")[1]
    if(item==""):

        item = item + " "
        pass

    label += [item]
    pass

annotation['label'] = label

##  Save the annotation.
path = "SOURCE/CSV/ANNOTATION.csv"
os.makedirs(os.path.dirname(path), exist_ok=True)
annotation.to_csv(path, index=False)


