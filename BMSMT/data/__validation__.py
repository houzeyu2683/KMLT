

##
##  Packages.  
import numpy
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


##
##  Class for validation.
class validation:

    def split(table, classification=None, ratio=0.2):

        numpy.random.seed(0)
        if(classification):

            train, check = train_test_split(table, stratify=table[classification], test_size=ratio)
            pass

        else:

            train, check = train_test_split(table, test_size=ratio)
            pass
        
        output = {"table":train}, {"table":check}
        return(output)

    # def fold(table, classification=None, size=4):

    #     numpy.random.seed(0)
    #     if(classification):

    #         validator  = StratifiedKFold(n_splits=size).split(table, table[classification])
    #         pass

    #     else:

    #         validator  = KFold(n_splits=size).split(table)
    #         pass

    #     output     = {}
    #     for number, index in enumerate(validator):

    #         group = {
    #             'train' : table.iloc[index[0]],
    #             'check' : table.iloc[index[1]]
    #         }        
    #         output[str(number+1)] = group
    #         pass

    #     return(output)
