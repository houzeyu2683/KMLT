
# import string

# item = "Ch/C6H5Cl4Sr32"

# #[i in string.ascii_lowercase for i in list(item)]

# loop = enumerate(list(item))

# output = []
# for index, letter in loop:
    
#     if(letter in string.ascii_lowercase):

#         output[-1] = output[-1] + letter
#         continue

#     if(letter in string.digits):

#         if(output[-1] in string.digits):

#             output[-1] = output[-1] + letter
#             continue

#         pass

#     output = output + [letter]




    


# ##
# ##  Packages.
# import torch, pickle, numpy


# ##
# ##  Dictionary, high level customization, case by case.
# path = "SOURCE/PICKLE/DICTIONARY.pickle"
# with open(path, 'rb') as paper:

#     dictionary = pickle.load(paper)
#     pass


# ##
# ##  Class for process target, case by case.
# class target:

#     def learn(item):

#         length  = 512
#         padding = 0
#         value   = [dictionary['index'][i] for i in list(item)]
#         index   = value + [padding] * (length-len(value))
#         # torch.argmax(torch.tensor(code), 0)
#         # code    = numpy.concatenate([numpy.expand_dims(projection[:,i], axis=1) for i in index], axis=1)
#         output  = torch.tensor(index, dtype=torch.long)
#         return(output)

#     def review(item):

#         length  = 512
#         padding = 0
#         value   = [dictionary['index'][i] for i in list(item)]
#         index   = value + [padding] * (length-len(value))
#         output  = torch.tensor(index, dtype=torch.long)
#         return(output)

#     # def convert(batch):

#     #     output = []
#     #     for index in batch:

#     #         key  = list(dictionary['index'].keys())
#     #         text = [key[i] for i in index]
#     #         output += [text]
#     #         pass

#     #     return(output)

# # [[1,2,3], [5,1,2], [4,4,4]]


# # index = [9,5,27]



# # [ for i in index]
# # [for k,v in dictionary.items()]




# # list(dictionary['index'].values())[3]

# # list(dictionary.values())
# # list(dictionary.keys())


# # dictionary.keys()
