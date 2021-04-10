

##
##  Packages.
import os, pandas, json


##
##  The [report] class.
class report:

    def __init__(self, train=None, check=None, test=None, folder="LOG"):

        self.train  = train
        self.check  = check
        self.test   = test
        self.folder = folder
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

        self.summary = pandas.concat([self.train, self.check, self.test], axis=1)
        pass

    def save(self):

        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, 'summary.csv')
        self.summary.to_csv(path, index=False)
        pass

##
##  The [write] function for saving the content to file.
# def write(content, folder, name):

#     with open(os.path.join(folder, name), "w+") as paper:
#         json.dump(content, paper)
#         pass


