
##
import os,pandas, pickle, tqdm

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
with open("SOURCE/LOG/test.pickle", "rb") as paper:

    paper = pickle.load(paper)
    pass

##
likelihood = pandas.DataFrame(paper['test']['likelihood'])
key = {
    "order":['$C', '$Cl', '$N', '$Si', '$P', '$Br', '$I', '$S', '$F', '$B', '$O', '$H'],
    "count":['#C', '#Cl', '#N', '#Si', '#P', '#Br', '#I', '#S', '#F', '#B', '#O', '#H']
}
likelihood.columns = key["order"] + key["count"]

##
text = ["InChI="+convert(item) for _, item in tqdm.tqdm(likelihood.iterrows(), total=len(likelihood), leave=False)]

##
label = pandas.read_csv("../DATA/BMSMT/TEST/CSV/LABEL.csv")
label['InChI'] = text
label.to_csv("SOURCE/LOG/test.csv", index=False)

min([len(i) for i in text])
[i for i in text if len(i)==16]
# item = paper['test']['likelihood'].iloc[0,:]
##
# test = {
#     "table":{
#         "likelihood":pandas.Dataframe(paper['test']['likelihood']),
#         "label":pandas.read_csv("../DATA/BMSMT/TEST/CSV/LABEL.csv")
#     }
# }

##

# test['table']['likelihood'].columns = key[order] + key[count]

# ##
# test['table']['label']['InChI'] = [convert(item) for item in test['table']['likelihood']]
# test['table']['label'].to_csv("",index=False)


