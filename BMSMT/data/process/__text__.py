

##
##
import string


##
##
def tokenize(item):

    loop = enumerate(list(item))

    token = []
    for _, letter in loop:
            
        if(letter in string.ascii_lowercase):

            if(token==[]):
                    
                token = token + [letter]
                pass

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
        
    output = token
    return(output)


##
##
class text:

    def learn(item):

        output = tokenize(item)
        return(output)

    def review(item):

        output = tokenize(item)
        return(output)






# item = "unknwon"
# index = text.tokenize(item)


#[i in string.ascii_lowercase for i in list(item)]
