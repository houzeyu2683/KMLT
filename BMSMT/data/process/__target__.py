

##
##  Packages.
import torch, pickle, numpy


##
##  Dictionary, high level customization, case by case.
path = "SOURCE/PICKLE/DICTIONARY.pickle"
with open(path, 'rb') as paper:

    dictionary = pickle.load(paper)
    pass


##
##  Class for process target, case by case.
class target:

    def learn(item):

        length  = 512
        padding = 0
        value   = [dictionary['index'][i] for i in list(item)]
        index   = value + [padding] * (length-len(value))
        # torch.argmax(torch.tensor(code), 0)
        # code    = numpy.concatenate([numpy.expand_dims(projection[:,i], axis=1) for i in index], axis=1)
        output  = torch.tensor(index, dtype=torch.long)
        return(output)

    def review(item):

        length  = 512
        padding = 0
        value   = [dictionary['index'][i] for i in list(item)]
        index   = value + [padding] * (length-len(value))
        # code    = numpy.concatenate([numpy.expand_dims(projection[:,i], axis=1) for i in index], axis=1)
        output  = torch.tensor(index, dtype=torch.long)
        return(output)

    def convert(batch):

        output = []
        for index in batch:

            key  = list(dictionary['index'].keys())
            text = [key[i] for i in index]
            output += [text]
            pass

        return(output)

# [[1,2,3], [5,1,2], [4,4,4]]


# index = [9,5,27]



# [ for i in index]
# [for k,v in dictionary.items()]




# list(dictionary['index'].values())[3]

# list(dictionary.values())
# list(dictionary.keys())


# dictionary.keys()
