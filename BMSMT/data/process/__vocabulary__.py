

##
##
import pickle, torch

##
##
class vocabulary:

    def load(path):

        with open(path, 'rb') as paper:

            output = pickle.load(paper)
            pass

        return(output)
    # def convert(self, text):

    #     index = [] 
    #     for item in text:

    #         item = [self.vocabulary['<bos>']] + [self.vocabulary[i] for i in item] + [self.vocabulary['<eos>']]
    #         item = torch.tensor(item, dtype=torch.long)
    #         index += [item]
    #         pass

    #     index = torch.nn.utils.rnn.pad_sequence(index, padding_value=self.vocabulary['<pad>'])
    #     return(index)
    #     pass
    # def load(path):

    #     with open(path, 'rb') as paper:

    #         output = pickle.load(paper)

    #     return(output)

