

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
##  Projection, high level customization, case by case.
## projection = numpy.eye(len(dictionary), k=0)


##
##  Class for process target, case by case.
class target:

    def learn(item):

        length  = 512
        padding = 0
        value   = [dictionary[i] for i in list(item)]
        index   = value + [padding] * (length-len(value))
        # torch.argmax(torch.tensor(code), 0)
        # code    = numpy.concatenate([numpy.expand_dims(projection[:,i], axis=1) for i in index], axis=1)
        output  = torch.tensor(index, dtype=torch.long)
        return(output)

    def review(item):

        length  = 512
        padding = 0
        value   = [dictionary[i] for i in list(item)]
        index   = value + [padding] * (length-len(value))
        # code    = numpy.concatenate([numpy.expand_dims(projection[:,i], axis=1) for i in index], axis=1)
        output  = torch.tensor(index, dtype=torch.long)
        return(output)

