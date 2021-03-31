
##  Packages.
import os, pandas, re, numpy, tqdm

##  Capture element function.
def capture(text):
    element = []
    for index, letter in enumerate(text):
        if( re.compile("[A-Z]").search(letter) ):
            element += [letter]
        if( re.compile("[a-z]").search(letter) ):
            element[-1] = element[-1] + letter
    return(element)

##  Load data.
data    = pandas.read_csv("DATA/BMSMT/TRAIN/CSV/LABEL.csv")

##  Get the element from data.
element = []
for index, row in tqdm.tqdm(data.iterrows(), total=len(data)):
    text = row['InChI'].split("InChI=1S/")[1].split('/')[0]
    element += capture(text)
element =pandas.DataFrame(set(element))

##  Summarize and save to file.
folder = "BMSMT/media"
os.makedirs(folder, exist_ok=True)
element.to_csv(os.path.join(folder, 'element.csv'), index=False, header=False)
