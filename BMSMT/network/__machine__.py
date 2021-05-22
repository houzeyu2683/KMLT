

##
##  Packages.
import os, tqdm, torch, numpy, pickle


##
##  Class for machine learning process, case by case.
class machine:

    def __init__(self, model, optimizer=None, criterion=None, device='cuda', folder=None, checkpoint="0"):

        self.model      = model
        self.optimizer  = optimizer
        self.criterion  = criterion
        self.device     = device
        self.folder     = folder
        self.checkpoint = checkpoint
        pass

        ##  Create the folder for storage.
        if(self.folder):
        
            os.makedirs(self.folder, exist_ok=True)
            pass
        
        ##  Optimizer schedule.
        if(self.optimizer):

            self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5, last_epoch=-1, verbose=False)
            pass

    def learn(self, train):

        ##  Mode of learn.
        self.model.train()
        self.model = self.model.to(self.device)
        pass

        for image, text in tqdm.tqdm(train, leave=False):

            ##  Handle batch.
            text   = self.model.convert(text).to(self.device)
            image  = image.to(self.device)
            batch  = image, text[:-1,:]

            ##  Update weight.
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss   = self.criterion.to(self.device)(output.flatten(0,1), text[1:,:].flatten())
            loss.backward()
            self.optimizer.step()
            pass
        
        print("End of epoch.")
        pass

    def measure(self, train=None, check=None):

        ##  Measurement.
        measurement = {}

        ##  Mode of evaluation.
        self.model.eval()
        self.model = self.model.to(self.device)

        ##  Event.
        event = {'train':train, "check":check}
        for key in event:

            if(event[key]):

                evaluation = {
                    'cost':[]
                }
                for image, text in tqdm.tqdm(event[key], leave=False):

                    ##  Handle batch.
                    image, text = image.to(self.device), self.model.convert(text).to(self.device)
                    batch = image, text[:-1,:]

                    ##  Evaluate item.
                    item = {
                        "likelihood":None,
                        "target":None
                    }
                    item["likelihood"] = self.model(batch).flatten(0,1)
                    item['target']     = text[1:,:].flatten()
                    cost = self.criterion(item['likelihood'], item['target']).cpu().detach().numpy().item()
                    evaluation['cost']  += [cost]
                    pass
                
                ##  Summarize evaluation.
                evaluation['cost'] = numpy.mean(evaluation['cost'])
                
                pass

                ##  Insert evaluation to measurement.
                measurement[key] = evaluation
                print("End of measure the {}.".format(key))
                pass

        self.measurement = measurement
        pass

    # def predict(self, test, length):

    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass

    #     prediction = []
    #     for batch in tqdm.tqdm(test, leave=False):

    #         image, _ = batch
    #         image = image.to(self.device)
    #         prediction += self.model.convert(image, length)
    #         pass
        
    #     return(prediction)

    def save(self, what='checkpoint'):

        ##  Save the checkpoint.
        if(what=='checkpoint'):

            path = os.path.join(self.folder, self.checkpoint+".checkpoint")
            torch.save(self.model.state_dict(), path)
            print("Save the weight of model to {}.".format(path))
            pass

        ##  Save the measurement.
        if(what=='measurement'):    

            path = os.path.join(self.folder, self.checkpoint + ".measurement")
            with open(path, 'wb') as paper:

                pickle.dump(self.measurement, paper)
                pass

            print("Save the checkpoint of measurement to {}.".format(path))
            pass
  
    def update(self, what='checkpoint'):

        if(what=='checkpoint'):
            
            try:
                
                self.checkpoint = str(int(self.checkpoint) + 1)
                print("Update the checkpoint to {} for next iteration.\n".format(self.checkpoint))
                pass

            except:

                print("The checkpoint is not integer, skip update checkpoint.\n")
                pass

        if(what=='schedule'):

            self.schedule.step()
            rate = self.optimizer.param_groups[0]['lr']
            print("The rate of optimizer is {} in the next loop.".format(rate))
            pass

    def load(self, what='weight', path=None):

        if(path):
            
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print("Finish loading.")
            return
        
        print("Fail loading.")
        pass



# import torch
# import torch.nn as nn
# loss = nn.CrossEntropyLoss()

# x = torch.randn((2,6,3))
# y = torch.randint(0,3,(2,6))
# x.shape
# y.shape
# loss(x[0,:,:], y[0,:])

# x[0,:,:]
# output.shape
# target = target.to('cuda')
# torch.flatten(target).shape
# loss   = criterion.to('cuda')(output, torch.flatten(target))