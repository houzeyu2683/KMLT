

##
##  Packages.
import os, tqdm, torch, numpy, pickle


##
##  The [machine] class.
class machine:

    def __init__(self, model, optimizer=None, criterion=None, device='cuda', folder="LOG", checkpoint="0"):

        self.model      = model
        self.optimizer  = optimizer
        self.criterion  = criterion
        self.device     = device
        self.folder     = folder
        self.checkpoint = checkpoint
        pass

        ##  Create the checkpoint folder.
        os.makedirs(self.folder, exist_ok=True)
        pass

    def learn(self, train):

        ##  Mode of learn.
        self.model.train()
        self.model = self.model.to(self.device)
        pass

        for batch in tqdm.tqdm(train, leave=False):

            ##  Process batch.
            image  = batch['image'].to(self.device)
            target = batch['target'].to(self.device)

            ##  Process feature and target.
            feature = image
            target  = target

            ##  Update weight.
            self.optimizer.zero_grad()
            output = self.model(feature)
            loss   = self.criterion.to(self.device)(output, target)
            loss.backward()
            self.optimizer.step()
            pass
        
        ##  Save model by folder and checkpoint.
        path = os.path.join(self.folder, self.checkpoint)
        print("Save the checkpoint of model to {}.".format(path))
        torch.save(self.model.state_dict(), path)
        pass

    def measure(self, train=None, check=None, test=None):

        ##  
        measurement = {}

        ##  Mode of evaluation.
        self.model.eval()
        self.model = self.model.to(self.device)

        ##
        event = {'train':train, "check":check, "test":test}
        for key in event:

            if(event[key]):

                item = {
                    'likelihood'    :[],
                    'target'        :[]
                }
                for batch in event[key]:

                    ##  Process batch.
                    image  = batch['image'].to(self.device)
                    target = batch['target'].to(self.device)

                    ##  Process feature and target.
                    feature = image
                    target  = target

                    ##  Prediction.
                    likelihood = self.model(feature)
                    prediction = likelihood
                    item['likelihood']  += [likelihood.cpu().detach().numpy()]
                    item['target']      += [target.cpu().detach().numpy()]
                    pass

                item['likelihood']  = numpy.concatenate(item['likelihood'], axis=0)
                item['target']      = numpy.concatenate(item['target'], axis=0)
                pass

                ##  Insert measurement.
                measurement[key] = item
                print("End of measure the {}.".format(key))
                pass
        
        path = os.path.join(self.folder, self.checkpoint + ".pickle")
        print("Save the checkpoint of measurement to {}.".format(path))
        with open(path, 'wb') as paper:

            pickle.dump(measurement, paper)
            pass
        
        self.measurement = measurement
        pass

    ##  
    def update(self, what='checkpoint'):

        if(what=='checkpoint'):
            
            self.checkpoint = str(int(self.checkpoint) + 1)
            print("Update the checkpoint to {} for next iteration.".format(self.checkpoint))
            pass
