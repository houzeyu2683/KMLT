
##  Packages.
import pandas, os, tqdm, string, pickle

##  Load annotation.
annotation = pandas.read_csv("SOURCE/CSV/ANNOTATION.csv")

##  Index of dictionary.
letter    = list(string.ascii_letters + string.punctuation + string.digits)
dictionary = {
    "index":{}
}
group = annotation.loc[annotation['mode']=='train']['label']
for index, alphabet in enumerate(letter):
    
    dictionary['index'].update({alphabet:index})
    pass

##  Save the dictionary.
path = "SOURCE/PICKLE/DICTIONARY.pickle"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "wb") as paper:

    pickle.dump(dictionary, paper)
    pass


