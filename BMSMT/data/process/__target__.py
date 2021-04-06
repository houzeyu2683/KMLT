

##
##  Packages.
import torch


##
##  The [target] class.
class target:

    def learn(item):

        key = [
            '$C', '$Cl', '$N', '$Si', '$P', '$Br', '$I', '$S', '$F', '$B', '$O', '$H',
            '#C', '#Cl', '#N', '#Si', '#P', '#Br', '#I', '#S', '#F', '#B', '#O', '#H'
        ]        
        output = torch.tensor(item[key], dtype=torch.long)
        return(output)

    def review(item):

        key = [
            '$C', '$Cl', '$N', '$Si', '$P', '$Br', '$I', '$S', '$F', '$B', '$O', '$H',
            '#C', '#Cl', '#N', '#Si', '#P', '#Br', '#I', '#S', '#F', '#B', '#O', '#H'
        ]        
        output = torch.tensor(item[key], dtype=torch.long)
        return(output)

