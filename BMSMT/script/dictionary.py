
##
import pandas, os, tqdm, string, pickle

##
annotation = pandas.read_csv("SOURCE/CSV/ANNOTATION.csv")

##
character = list(string.ascii_letters + string.punctuation + string.digits)
character.remove(".")

##
dictionary = {".":0}
for index, item in enumerate(character, 1):
    
    dictionary.update({item:index})
    pass


##
path = "SOURCE/PICKLE/DICTIONARY.pickle"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "wb") as paper:

    pickle.dump(dictionary, paper)
    pass

