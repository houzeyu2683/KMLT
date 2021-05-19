
##  Packages.
import pandas, os, tqdm, string, pickle, collections, torchtext

##  Load annotation.
annotation = pandas.read_csv("SOURCE/CSV/ANNOTATION.csv")

##
def tokenize(item):

    loop = enumerate(list(item))

    output = []
    for _, letter in loop:
            
        if(letter in string.ascii_lowercase):

            if(not output[-1] in string.punctuation):
                
                output[-1] = output[-1] + letter
                continue

            pass

        if(letter in string.digits):

            if(output[-1] in string.digits):

                output[-1] = output[-1] + letter
                continue

            pass

        output = output + [letter]
        pass

    return(output)

##
count = collections.Counter()
for i in tqdm.tqdm(annotation[annotation['mode']=='train']['label']):

    count.update(tokenize(i))
    pass

##
vocabulary = torchtext.vocab.Vocab(count, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
path = "SOURCE/PICKLE/VOCABULARY.pickle"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "wb") as paper:

    pickle.dump(vocabulary, paper)
    pass





# count.update([""])

# ##  Index of dictionary.
# letter    = list(string.ascii_letters + string.punctuation + string.digits)
# dictionary = {
#     "index":{}
# }
# group = annotation.loc[annotation['mode']=='train']['label']
# for index, alphabet in enumerate(letter):
    
#     dictionary['index'].update({alphabet:index})
#     pass

##  Save the dictionary.
# path = "SOURCE/PICKLE/DICTIONARY.pickle"
# os.makedirs(os.path.dirname(path), exist_ok=True)
# with open(path, "wb") as paper:

#     pickle.dump(dictionary, paper)
#     pass


