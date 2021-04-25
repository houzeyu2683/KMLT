

##
##  Packages.
import os, tqdm, torch, numpy, pickle


##
##  Class for machine learning process, case by case.
class machine:

    def __init__(self, model, optimizer=None, criterion=None, device='cuda', folder="LOG", checkpoint="0"):

        self.model      = model
        self.optimizer  = optimizer
        self.criterion  = criterion
        self.device     = device
        self.folder     = folder
        self.checkpoint = checkpoint
        pass

        ##  Create the folder for checkpoint and measurenent.
        os.makedirs(self.folder, exist_ok=True)
        pass

    def learn(self, train):

        ##  Mode of learn.
        self.model.train()
        self.model = self.model.to(self.device)
        pass

        for batch in tqdm.tqdm(train, leave=False):

            ##  Handle batch.
            feature, target = batch
            feature, target = feature.to(self.device), target.to(self.device)
            batch = feature, target

            ##  Update weight.
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss   = self.criterion.to(self.device)(output[0], output[1])
            loss.backward()
            self.optimizer.step()
            pass

    def measure(self, train=None, check=None, test=None):

        ##  Measurement.
        measurement = {}

        ##  Mode of evaluation.
        self.model.eval()
        self.model = self.model.to(self.device)

        ##  Event.
        event = {'train':train, "check":check, "test":test}
        for key in event:

            if(event[key]):

                item = {
                    'likelihood'    :[],
                    'target'        :[]
                }
                for batch in tqdm.tqdm(event[key], leave=False):

                    ##  Handle batch.
                    feature, target = batch
                    feature, target = feature.to(self.device), target.to(self.device)
                    batch = feature, target

                    ##  Evaluation.
                    likelihood, target = self.model(batch)
                    item['likelihood']  += [likelihood.detach().cpu()]
                    item['target']      += [target.detach().cpu()]
                    pass
                
                item['likelihood'] = torch.cat(item['likelihood'], dim=0).numpy()
                item['target']     = torch.cat(item['target'], dim=0).numpy()
                pass

                ##  Insert measurement.
                measurement[key] = item
                print("End of measure the {}.".format(key))
                pass
        
        self.measurement = measurement
        pass

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
                print("Update the checkpoint to {} for next iteration.".format(self.checkpoint))
                pass

            except:

                print("The checkpoint is not integer, ignore update checkpoint.")
                pass


