
##
import os,pandas, pickle

##
def convert(item):

    ##  Order
    order = item[['$' in key for key in item.keys()]]
    order = order[order>0].sort_values()
    order = [key.split('$')[1] for key in order.keys()]

    ##  Count
    count = item[['#' in key for key in item.keys()]]
    count = count[[key.split('#')[1] in order for key in count.keys()]]
    count = round(count).astype('int').astype('str')
    count.index = [key.split("#")[1] for key in count.keys()]


    ##  
    text = ""
    for key, value in count[order].to_dict().items():

        text += key + value
        pass

    return(text)


##
with open("", "rb") as paper:

    paper = pickle.load(paper)
    pass

##
test = {
    "table":{
        "likelihood":pandas.Dataframe(paper['test']['likelihood']),
        "label":pandas.read_csv("../DATA/BMSMT/TEST/CSV/LABEL.csv")
    }
}

##
key = {
    "order":['$C', '$Cl', '$N', '$Si', '$P', '$Br', '$I', '$S', '$F', '$B', '$O', '$H'],
    "count":['#C', '#Cl', '#N', '#Si', '#P', '#Br', '#I', '#S', '#F', '#B', '#O', '#H']
}
test['table']['likelihood'].columns = key[order] + key[count]

##
test['table']['label']['InChI'] = [convert(item) for item in test['table']['likelihood']]
test['table']['label'].to_csv("",index=False)


