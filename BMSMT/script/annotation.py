
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

##  Check the, "InChI" column.
##  I make sure the symbol '.' does not exist in this column,
##  then define the '.' symbol is the padding of sequence.
##  I make sure the max length of sequence is 403,
##  then define the length of sequence is 512.
##  Create the "label" column.
annotation['label'] = [i.split("InChI=1S/")[1] for i in annotation['InChI']]
# length = 512
# annotation['label'] = annotation['label'].str.pad(width=length, side='right', fillchar='.')

##  Save the annotation.
path = "SOURCE/CSV/ANNOTATION.csv"
os.makedirs(os.path.dirname(path), exist_ok=True)
annotation.to_csv(path, index=False)

