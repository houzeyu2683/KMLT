
##  Packages.
import pandas, os, tqdm, string, pickle

##  Load annotation.
annotation = pandas.read_csv("SOURCE/CSV/ANNOTATION.csv")

##
letter    = list(string.ascii_letters + string.punctuation + string.digits)
dictionary = {
    "character":{},
    "weight":[]
}
group = annotation.loc[annotation['mode']=='train']['InChI']
for index, alphabet in enumerate(letter):
    
    weight = sum([i.count(alphabet) for i in tqdm.tqdm(group, leave=False)])
    if(weight==0):

        weight = 1e-2
        pass
    dictionary['character'].update({alphabet:index})
    dictionary['weight'] += [float(weight)]
    pass

##  Scale the weight.
dictionary['weight'] = [i / max(dictionary['weight']) for i in dictionary['weight']]

##
path = "SOURCE/PICKLE/DICTIONARY.pickle"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "wb") as paper:

    pickle.dump(dictionary, paper)
    pass

dictionary
##
# dictionary['weight'].values()

# w = '.'


# annotation.loc[annotation['mode']=='train']['InChI'].head().tolist()



# group = {
#     "word list" : []
# }
# for i in tqdm.tqdm(annotation.loc[annotation['mode']=='train']['InChI'], leave=False):

#     group['word list'] += list(i)
#     pass

# group['word list']
# [list(i) for i in annotation.loc[annotation['mode']=='train']['InChI'].head()]

# ##
# character = {}
# for i in list(string.ascii_letters + string.punctuation + string.digits):


#     break



# ##  According to train data,
# ##  Check the target, "InChI" column.
# ##  I make sure the symbol '.' does not exist in this column.
# ##  Define the '.' symbol is the padding of sequence.
# ##  The max length of sequence is 403.
# ##  Define the length of sequence is 512.


# # length = 512
# # for index, item in tqdm.tqdm(annotation.iterrows(), total=len(annotation), leave=False):

# #     label = item['InChI']
# #     padding = 512 - len(label)
# #     annotation._set_value(index, "InChI", label + ('.' * padding))
# #     pass    

# ##
# character = list(string.ascii_letters + string.punctuation + string.digits)
# character.remove(".")

# ##
# dictionary = {".":0}
# for index, item in enumerate(character, 1):
    
#     dictionary.update({item:index})
#     pass





# ##===========##

# ##
# import pandas, os, tqdm, string, pickle, itertools

# ##
# annotation = pandas.read_csv("SOURCE/CSV/ANNOTATION.csv")

# ##
# train = {}
# train['table']              = annotation.loc[annotation['mode']=='train']
# train['label word list']    = []
# for i in tqdm.tqdm(train['table']['InChI'], leave=False):

#     train['label word list'] += list(i)
#     pass

# character = list(string.ascii_letters + string.punctuation + string.digits)

# train['label word list'].count("C")

# train['label word list']    = [list(i) for i in tqdm.tqdm(train['table']['InChI'], leave=False)]
# #train['label word group']   = list(itertools.chain(*train['label word list']))
# train['label unique word']  = set(train['label word group'])
# if("." not in train['label unique word']):

#     print("The symbol '.' does not exist.")
#     pass



# list.count()

# print(*['a','b'])
# print(['a','b'])
# list(annotation['InChI'][0])



# List_2D = [[1,2,3],[4,5,6],[7,8,9]]

# List_flat = list(itertools.chain(*train['word list']))

