
##
import os, pandas, json

##
class history:

    def __init__(self, train=None, check=None, test=None):

        self.train = train
        self.check = check
        self.test  = test
        pass
    
    def summarize(self):

        if(self.train):

            self.train = pandas.DataFrame(self.train)
            self.train.columns = ["train " + i for i in self.train.columns.tolist()]
            pass

        if(self.check):

            self.check = pandas.DataFrame(self.check)
            self.check.columns = ["check " + i for i in self.check.columns.tolist()]
            pass

        if(self.test):
            
            self.test  = pandas.DataFrame(self.test)
            self.test.columns  = ["test "  + i for i in self.test.columns.tolist() ]
            pass        

        ##  Convert to table.
        self.summary = pandas.concat([self.train, self.check, self.test], axis=1)
        pass

    ##  
    def save(self, folder, name='summary.csv'):

        os.makedirs(folder, exist_ok=True)
        self.summary.to_csv(os.path.join(folder, name), index=False)
        pass

##
##  The [write] function for saving the content to file.
# def write(content, folder, name):

#     with open(os.path.join(folder, name), "w+") as paper:
#         json.dump(content, paper)
#         pass


