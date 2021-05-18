

##
##
import torch, torchvision, pickle
import torch.nn as nn


##
##
path = "SOURCE/PICKLE/VOCABULARY.pickle"
with open(path, 'rb') as paper:

    vocabulary = pickle.load(paper)
    pass


class mask:

    def encode(text):

        if(text.is_cuda):

            device = "cuda"
            pass

        else:

            device = 'cpu'
            pass

        length = text.shape[0]
        mask = torch.zeros((length, length), device=device).type(torch.bool)
        return mask
    
    def decode(text):

        if(text.is_cuda):

            device = "cuda"
            pass

        else:

            device = 'cpu'
            pass

        length = text.shape[0]
        mask = (torch.triu(torch.ones((length, length), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def pad(text):

        mask = (text == vocabulary['<pad>']).transpose(0, 1)      
        return mask


class model(torch.nn.Module):

    def __init__(self):
        
        super(model, self).__init__()
        pass

        self.number = {
            "vocabulary" : 141,
            "embedding" : 256,
            'sequence' : 512
        }
        pass

        layer = {}
        layer['image to index 01'] = nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1], nn.Sigmoid())
        layer['image to index 02'] = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        layer['text encoder'] = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.number['embedding'], nhead=2), num_layers=2)
        layer['text decoder'] = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=self.number['embedding'], nhead=2), num_layers=2)
        layer['text to embedding'] = nn.Embedding(self.number['vocabulary'], self.number['embedding'])
        layer['text to vacabulary'] = nn.Linear(self.number['embedding'], self.number['vocabulary'])
        self.layer = nn.ModuleDict(layer)
        pass
    

    def forward(self, batch):
        
        image, text = batch
        midden = {}
        ##  Image to index of text, prototype.
        midden['image to index 01 (01)'] = self.layer['image to index 01'](image)
        midden['image to index 01 (02)'] = midden['image to index 01 (01)'].flatten(1,-1)
        midden['image to index 01 (03)'] = torch.as_tensor(midden['image to index 01 (02)']*(self.number['vocabulary']), dtype=torch.long)
        
        ##  Image to index of text, given special tag.
        midden['image to index 02 (01)'] = self.layer['image to index 02'](midden['image to index 01 (02)'])
        midden['image to index 02 (02)'] = torch.as_tensor(midden['image to index 02 (01)'] * (self.number['sequence']-3), dtype=torch.long)
        for row, column in enumerate(midden['image to index 02 (02)']):

            midden['image to index 01 (03)'][row, 0] = vocabulary['<bos>']
            midden['image to index 01 (03)'][row, column] = vocabulary['<eos>']
            midden['image to index 01 (03)'][row, column+1:] = vocabulary['<pad>']
            pass    
        
        ##  Encoder, index of text to encode.
        midden['image to index 03'] = midden['image to index 01 (03)'].transpose(0,1)
        midden['encoder memory'] = self.layer['text encoder'](
            self.layer['text to embedding'](midden['image to index 03']),
            mask.encode(midden['image to index 03']), 
            mask.pad(midden['image to index 03'])
        )

        ##  Decoder, encode to index of text.
        midden['decoder output'] = self.layer['text decoder'](
            self.layer['text to embedding'](text), 
            midden['encoder memory'], 
            mask.decode(text), 
            None, 
            mask.pad(text), 
            None
        )
        output = self.layer['text to vacabulary'](midden['decoder output'])
        # print("self.generator(outs)-----")
        # print(self.generator(outs).shape)
        return output


    def convert(self, image, length):

        batch = image.shape[0]

        if(image.is_cuda):

            device = 'cuda'
            pass

        else:
     
            device = 'cpu'
            pass

        midden = {}
        ##  Image to index of text, prototype.
        midden['image to index 01 (01)'] = self.layer['image to index 01'](image)
        midden['image to index 01 (02)'] = midden['image to index 01 (01)'].flatten(1,-1)
        midden['image to index 01 (03)'] = torch.as_tensor(midden['image to index 01 (02)']*(self.number['vocabulary']), dtype=torch.long)

        ##  Image to index of text, given special tag.
        midden['image to index 02 (01)'] = self.layer['image to index 02'](midden['image to index 01 (02)'])
        midden['image to index 02 (02)'] = torch.as_tensor(midden['image to index 02 (01)'] * (self.number['sequence']-3), dtype=torch.long)
        for row, column in enumerate(midden['image to index 02 (02)']):

            midden['image to index 01 (03)'][row, 0] = vocabulary['<bos>']
            midden['image to index 01 (03)'][row, column] = vocabulary['<eos>']
            midden['image to index 01 (03)'][row, column+1:] = vocabulary['<pad>']
            pass    

        ##  Encoder, index of text to encode.
        midden['image to index 03'] = midden['image to index 01 (03)'].transpose(0,1)
        midden['encoder memory'] = self.layer['text encoder'](
            self.layer['text to embedding'](midden['image to index 03']),
            mask.encode(midden['image to index 03']), 
            mask.pad(midden['image to index 03'])
        )

        ##
        memory = midden['encoder memory']
        # print(memory.shape)
        sequence = torch.ones(1, batch).fill_(vocabulary['<bos>']).type(torch.long).to(device)
        # print("sequence")
        # print(sequence.shape)
        for _ in range(length):
            midden['decoder output'] = self.layer['text decoder'](
                self.layer['text to embedding'](sequence), 
                memory, 
                mask.decode(sequence), 
                None, 
                None
            )
            # print("midden['decoder output']-----")
            # print(midden['decoder output'].shape)
            probability = self.layer['text to vacabulary'](midden['decoder output'].transpose(0, 1)[:, -1])
            # print("probability---")
            # print(probability.shape)
            _, prediction = torch.max(probability, dim = 1)
            # print('prediction')
            # print(prediction.shape)
            # print(prediction)
            sequence = torch.cat([sequence, prediction.unsqueeze(dim=0)], dim=0)
            # print("sequence")
            # print(sequence.shape)
            output = []
            for i in range(batch):

                character = "InChI=1S/" + "".join([vocabulary.itos[token] for token in sequence[:,i]]).replace("<bos>", "").replace("<eos>", "").replace('<pad>', "")
                output += [character]
                pass

        return output

        output = []
        for item in range(batch):
            
            memory = midden['encoder memory'][:,item:item+1,:]

        # print("midden['encoder memory']")
        # print(midden['encoder memory'].shape)
            ##  Generate sequence.
            sequence = torch.ones(1, 1).fill_(vocabulary['<bos>']).type(torch.long).to(device)
            for i in range(length):

                midden['decoder output'] = self.layer['text decoder'](
                    self.layer['text to embedding'](sequence), 
                    memory, 
                    mask.decode(sequence), 
                    None, 
                    None
                )
                print("midden['decoder output'] ")
                print(midden['decoder output'].shape)
                probability = self.layer['text to vacabulary'](midden['decoder output'].transpose(0, 1)[:, -1])
                _, prediction = torch.max(probability, dim = 1)
                index = prediction.item()
                sequence = torch.cat([sequence, torch.ones(1, 1).type_as(midden['image to index 03']).fill_(index)], dim=0)
                pass

                if index == vocabulary['<eos>']:
                    
                    break
            
            character = "InChI=1S/" + "".join([vocabulary.itos[tok] for tok in sequence]).replace("<bos>", "").replace("<eos>", "")
            output += [character]
            pass

        return output




# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     src = src.to(device)
#     src_mask = src_mask.to(device)

#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
#     for i in range(max_len-1):
#         memory = memory.to(device)
#         memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)

#         tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
#         print('tgt_mask----')
#         print(tgt_mask)
#         print(tgt_mask.shape)


#         out = model.decode(ys, memory, tgt_mask)
#         out = out.transpose(0, 1)
#         print("output===")
#         print(out.shape)
#         print(out)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim = 1)
#         next_word = next_word.item()

#         ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=0)
#         if next_word == EOS_IDX:
#           break
#     return ys

# def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
#     model.eval()
#     tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)]+ [EOS_IDX]
#     num_tokens = len(tokens)
#     src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
#     return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")

# # batch = torch.randn((8, 3, 224, 224)), torch.randint(0, 141, (10, 8))
# # image, text = batch
# # m = model()
# # x = m(batch)

# # x.shape