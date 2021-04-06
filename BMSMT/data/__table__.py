

##
##  Packages.
import pandas


##
##  The [table] class.
class table:

    def read(path):

        output = pandas.read_csv(path)
        return(output)

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
