

##
##
import string, torch, pickle


# ##
# ##
# path = "SOURCE/PICKLE/VOCABULARY.pickle"
# with open(path, 'rb') as paper:

#     vocabulary = pickle.load(paper)
#     pass


# ##
# ##
# length = 512


##
##
class text:

    def tokenize(item):

        loop = enumerate(list(item))

        token = []
        for _, letter in loop:
            
            if(letter in string.ascii_lowercase):

                if(not token[-1] in string.punctuation):
                
                    token[-1] = token[-1] + letter
                    continue

                pass

            if(letter in string.digits):

                if(token[-1] in string.digits):

                    token[-1] = token[-1] + letter
                    continue

                pass
            
            token = token + [letter]
            pass
        
        # index = [vocabulary['<bos>']] + [vocabulary[i] for i in token] + [vocabulary['<eos>']]
        # index = index + ([vocabulary['<pad>']] * (length - len(index)))
        # index = torch.tensor(index, dtype=torch.long)
        output = token
        return(output)



# item = "unknwon"
# index = text.tokenize(item)


#[i in string.ascii_lowercase for i in list(item)]
