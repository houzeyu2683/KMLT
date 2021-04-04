
##
import os, tqdm, torch, numpy

##
class machine:

    def __init__(self, model, optimizer=None, criterion=None, device='cuda', record=None, checkpoint=0):

        self.model      = model
        self.optimizer  = optimizer
        self.criterion  = criterion
        self.device     = device
        self.record     = record

        ##  The initial checkpoint.
        self.checkpoint = checkpoint
        if(self.checkpoint):
        
            print('The initial checkpoint is {}.'.format(self.checkpoint))
            pass

        ##  Create the folder.
        if(self.record):

            os.makedirs(self.record, exist_ok=True)
            pass

    def learn(self, train):

        ##  Mode of learn.
        self.model.train()
        self.model = self.model.to(self.device)
        for batch in tqdm.tqdm(train, leave=False):

            ##  Process batch.
            feature, target = batch
            image          = feature['image'].to(self.device)
            classification = target['classification'].to(self.device)

            ##  Process feature and target.
            feature = {"image":image}
            target  = {'classification':classification}

            ##  Update weights.
            self.optimizer.zero_grad()
            output = self.model(feature)
            loss   = self.criterion.to(self.device)(output, target['classification'])
            loss.backward()
            self.optimizer.step()
            pass
        
        ##  Save the model.
        torch.save(self.model.state_dict(), os.path.join(self.record, str(self.checkpoint)))
        print("Save the model to the checkpoint to {}.".format(self.checkpoint))

        ##  Update the checkpoint.
        self.checkpoint = self.checkpoint + 1
        print("Update the checkpoint to {} for next iteration.".format(self.checkpoint))
        pass

    def measure(self, train=None, check=None, test=None):

        ##  The [measurement] is a dictionary for saving the result.
        measurement = {}

        ##  Mode of evaluation.
        self.model.eval()
        self.model = self.model.to(self.device)

        ##  Measure the [train].
        if(train):
            
            item = {
                'likelihood'    :[],
                "prediction"    :[],
                'classification':[]
            }
            for batch in train:

                ##  Process batch.
                feature, target = batch
                image          = feature['image'].to(self.device)
                classification = target['classification'].to(self.device)

                ##  Process feature and target.
                feature = {"image":image}
                target  = {'classification':classification}

                ##  Prediction.
                likelihood = self.model(feature)
                prediction = torch.argmax(likelihood, dim=1)
                item['likelihood']     += [likelihood.cpu().detach().numpy()]
                item['prediction']     += prediction.cpu().detach().numpy().tolist()
                item['classification'] += target['classification'].cpu().detach().numpy().tolist()
                pass

            item['likelihood']     = numpy.concatenate(item['likelihood'], axis=0)
            item['prediction']     = item['prediction']
            item['classification'] = item['classification']

            ##  Insert [item] to [measurement].
            measurement['train'] = item
            print("End of measure the [train].")
            pass

        ##  Measure the [check].
        if(check):
            
            item = {
                'likelihood'    :[],
                "prediction"    :[],
                'classification':[]
            }
            for batch in check:
                
                ##  Process batch.
                feature, target = batch
                image          = feature['image'].to(self.device)
                classification = target['classification'].to(self.device)

                ##  Process feature and target.
                feature = {"image":image}
                target  = {'classification':classification}

                ##  Prediction.
                likelihood = self.model(feature)
                prediction = torch.argmax(likelihood, dim=1)
                item['likelihood']     += [likelihood.cpu().detach().numpy()]
                item['prediction']     += prediction.cpu().detach().numpy().tolist()
                item['classification'] += target['classification'].cpu().detach().numpy().tolist()
                pass

            item['likelihood']     = numpy.concatenate(item['likelihood'], axis=0)
            item['prediction']     = item['prediction']
            item['classification'] = item['classification']

            ##  Insert [item] to [measurement].
            measurement['check'] = item
            print("End of measure the [check].")
            pass

        ##  Measure the [test].
        if(test):

            item = {
                'likelihood'    :[],
                "prediction"    :[],
                'classification':[]
            }
            for batch in test:

                ##  Process batch.
                feature, target = batch
                image          = feature['image'].to(self.device)
                classification = target['classification'].to(self.device)

                ##  Process feature and target.
                feature = {"image":image}
                target  = {'classification':classification}

                ##  Prediction.
                likelihood = self.model(feature)
                prediction = torch.argmax(likelihood, dim=1)
                item['likelihood']     += [likelihood.cpu().detach().numpy()]
                item['prediction']     += prediction.cpu().detach().numpy().tolist()
                item['classification'] += target['classification'].cpu().detach().numpy().tolist()
                pass

            item['likelihood']     = numpy.concatenate(item['likelihood'], axis=0)
            item['prediction']     = item['prediction']
            item['classification'] = item['classification']

            ##  Insert [item] to [measurement].
            measurement['test'] = item
            print("End of measure the [test].")
            pass
        
        return(measurement)