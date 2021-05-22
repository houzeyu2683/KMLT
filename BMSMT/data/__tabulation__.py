

##
##  Packages.
import pandas


##
##  Class for tabulation.
class tabulation:
    
    def read(path, number=None):

        if(number):

            output = pandas.read_csv(path, dtype = str, nrows=number)
            pass

        else:

            output = pandas.read_csv(path, dtype = str)
            pass
        
        return(output)

    ##  Filter by column.
    def filter(table, column, value):

        output = table.loc[table[column]==value].copy()
        return(output)

    ##  Balance the data of table with target.
    def balance(table, target, size):

        output = []
        for i in set(table[target]):

            selection = table[table[target]==i]
            pass
        
            if(len(selection)>size):

                selection = selection.sample(size)
                pass

            else:

                selection = selection.sample(size, replace=True)
                pass

            output = output + [selection]
            pass

        output = pandas.concat(output, axis=0)
        return(output)

    ##
    def unbalance(table, target, size):

        group = []
        for key, value in size.items():

            selection = table.loc[table[target]==key]
            pass

            if(len(selection)>value):
        
                group += [selection.sample(value)]
                pass
        
            else:
        
                group += [selection.sample(value, replace=True)]
                pass
        
        output = pandas.concat(group, axis=0)
        return(output)
