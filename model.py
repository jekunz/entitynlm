import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import CorpusLoader

import torch.nn.init as init
import numpy as np
import time


class EntityNLM(nn.Module):
    def __init__(self, vocab_size, embedding_size=256, hidden_size=256, num_layers=1, dropout=0.5, train_states=None, tied_embedding=None):
        super(EntityNLM, self).__init__()

        # assert hidden_size == entity_size, "hidden_size should be equal to entity_size"
        # embedding matrix for input tokens
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_size)

        # LSTM
        self.lstm = nn.LSTM(input_size=embedding_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers )
        
        # final layer, outputs probability distribution for vocabulary
        self.output_layer = nn.Linear(hidden_size, vocab_size) 
        
        # r is the parameterized embedding associated with r, which paves the way for exploring entity type representations in future work
        self.r_embeddings = torch.nn.Parameter(torch.FloatTensor(2, hidden_size), requires_grad=True).to(device)
        # W_r is parameter matrix for the bilinear score for h_t−1 and r.
        self.W_r = nn.Bilinear(hidden_size, hidden_size, 1)
        
        
        # W_length is the weight matrix for length prediction
        self.W_length = nn.Linear(2 * hidden_size, 25)    
        
        # W_entity is the weight matrix for predicting entities using their continuous representations
        self.W_entity = nn.Bilinear(hidden_size, hidden_size, 1)

        # For distance feature
        self.w_dist = nn.Linear(1, 1)

        # matrix to create interpolation value δt
        self.W_delta = nn.Bilinear(hidden_size, hidden_size, 1)
        
        # W_e is a transformation matrix to adjust the dimensionality of e_current
        self.W_e = nn.Linear(hidden_size, hidden_size)
    
        # dropout layer
        self.dropout = nn.Dropout(dropout)
    
        # set of entities E_t
        self.entities      = torch.tensor([], dtype=torch.float, device=device)
        # distance features for entities
        self.dist_features = torch.tensor([], dtype=torch.float, device=device)
        self.max_entity_index = 0

        self.init_weights()
        
    def init_weights(self, init_range=(-0.01, 0.01)):
        if not init_range:
            return
        
        for param in self.parameters():
            if param.dim() > 1:
                init.xavier_uniform_(param,gain=np.sqrt(2))  
        
        self.W_entity.weight.data.uniform_(*init_range)
        self.W_entity.bias.data.fill_(0)
        
        self.w_dist.weight.data.uniform_(*init_range)
        self.w_dist.bias.data.fill_(0)
        
        self.W_e.weight.data.uniform_(*init_range)
        self.W_e.bias.data.fill_(0)
        
        
    def forward_rnn(self, x, states):
        # Input: LongTensor with token indices
        # Creates embedding vectors for input and feeds trough lstm
        x = self.embedding_matrix(x.view(1, -1))
        return self.lstm(x, states)
    
    def get_new_entity(self):
        # creates a new entity and returns the reference
        self.add_new_entity()
        return self.get_entity_embedding(-1)
    
    def initialize_entity_embedding(self, sigma=0.01):
        # get embedding for r == 1
        r1 = self.r_embeddings[1]
        # normal init based on r1 with sigma
        u = r1 + sigma*torch.normal(torch.zeros_like(r1, device=device), torch.ones_like(r1, device=device)).view(1,-1)
        # normalize embedding
        u = u / torch.norm(u, p = 2)
        return u
    
    def add_new_entity(self, t=0.0):
        # append a new entity including distance features
        # append new embedding u to entity matrix
        self.entities = torch.cat((self.entities, self.initialize_entity_embedding()), dim=0)
        # create respective distance features
        self.dist_features = torch.cat((self.dist_features, torch.tensor([[t]], dtype=torch.float, device=device)), dim=0)

    def update_entity_embedding(self, entity_index, h_t, t):
        # get entity
        entity_embedding = self.get_entity_embedding(entity_index)
        # calculate interpolation δ_t
        delta = torch.sigmoid(self.W_delta(entity_embedding, h_t)).view(-1)

        # update entity embedding based with h_t using δ_t
        updated_embedding = delta * entity_embedding + ( 1 - delta ) * h_t

        # update entities in set E_t
        self.entities      = self.entities.index_copy(0, torch.tensor(entity_index), (updated_embedding / torch.norm(updated_embedding)))
        # update entities in self.dist_features
        self.dist_features = self.dist_features.index_copy(0, torch.tensor(entity_index), torch.tensor([[t]], dtype=torch.float, device=device))
    
    def get_entity_embedding(self, entity_index):
        # returns the entity embedding to the respective index
        return self.entities[entity_index].unsqueeze(0)

    def get_dist_feat(self, t):
        # subtract current time step from dist feature vector
        return self.dist_features - t

    def get_next_R(self, h_t):
        # predict distribution for next R
        pred_r = self.W_r( self.dropout(self.r_embeddings), self.dropout(h_t.expand_as(self.r_embeddings)) ).view(1, -1)
        return pred_r

    def get_next_E(self, h_t, t):
        # predict next entity
        if self.max_entity_index == self.entities.size(0)-1: # max_entity_index is last element
            self.add_new_entity()                           # create new entity slot
        dist_feat = self.get_dist_feat(t)                    # distance features for current time step
        # apply bilinear W_entity
        pred_e = self.W_entity(self.dropout(self.entities), self.dropout(h_t.expand_as(self.entities)) ) + self.w_dist(self.dropout(dist_feat))
        return pred_e.view(1, -1)

    def get_next_L(self, h_t, entity_embedding):
        # predict length of next entity
        return self.W_length(self.dropout(torch.cat((h_t, entity_embedding),dim=1)))

    def get_next_X(self, h_t, e_current):
        # predict next token
        return self.output_layer(self.dropout(h_t + self.W_e(self.dropout(e_current))))
    
    def register_predicted_entity(self, e_index):
        # this function registers entities to determine 
        # if there is a free slot in the entitiy set
        new_max = max(int(e_index), self.max_entity_index)
        self.max_entity_index = new_max

    def reset_state(self):
        # reset all entity states
        self.entities      = torch.tensor([], dtype=torch.float, device=device)
        self.dist_features = torch.tensor([], dtype=torch.float, device=device)
        self.max_entity_index = 0


def run_nlm(model, corpus, optimizer=None, epochs=1, eval_corpus=None, status_interval=25, str_pattern='{}_{}_epoch_{}.pkl', rz_amplifier=1):
    entity_offset = 1

    for epoch in range(1, epochs+1):
        X_epoch_loss, E_epoch_loss, R_epoch_loss, L_epoch_loss = 0, 0, 0, 0
        epoch_tokens, epoch_r_div, epoch_l_div, epoch_e_div = 0, 0, 0, 0
        epoch_start = time.time()
        count_E = 0
        count_E_correct = 0
        count_R = 0
        r_true_positive  = 0
        r_false_positive = 0
        for i_doc, doc in enumerate(corpus.gen()):
            model.reset_state()
            # initialize e_current
            e_current = model.get_new_entity()
            
            # forward first token through Embedding and RNN
            # initialize states
            # lstm initializes states with zeros when given None
            h_t, states = model.forward_rnn(doc.X[0], states=None)
            h_t = h_t.squeeze(0)
            
            # initialize loss tensors
            X_loss = torch.tensor(0, dtype=torch.float, device=device)
            E_loss = torch.tensor(0, dtype=torch.float, device=device)
            R_loss = torch.tensor(0, dtype=torch.float, device=device)
            L_loss = torch.tensor(0, dtype=torch.float, device=device)
            
            # counters to properly devide losses
            r_div = 0
            l_div = 0
            e_div = 0

            # counter for stats
            doc_r_true_positive  = 0
            doc_r_false_positive = 0
            doc_count_R          = 0

            
            # iterate over document
            for t in range(doc.X.size(0)-1):      
                # define target values
                next_X = doc.X[t+1] # next Token
                next_E = doc.E[t+1] - entity_offset # next Entity, offset to match indices with self.entities
                next_R = doc.R[t+1] # next R type
                next_L = doc.L[t+1] # next Length
                
                # Start Paper
                # Define current value for L
                current_L = doc.L[t]
                if current_L == 1:
                    # 1. 
                    # last L equals 1, not continuing entity mention
                    
                    # predict next R
                    R_dist = model.get_next_R(h_t)
                    # create loss for R
                    r_current_loss = torch.nn.functional.cross_entropy(R_dist, next_R.view(-1))*rz_amplifier
                    # r_current_loss is used to make amplification of loss possible
                    R_loss += r_current_loss
                    
                    # add division counter for R loss
                    r_div  += 1

                    
                    if next_R == 1:
                        # next Token is within an entity mention
                        doc_count_R += 1
                        if R_dist.argmax():
                            # both True - correct pred
                            doc_r_true_positive  += 1
                            #R_loss += r_current_loss
                        else:
                            # false negative prediction
                            # extra loss
                            R_loss += r_current_loss * 10
                            pass

                        
                        # select the entity
                        E_dist = model.get_next_E(h_t, t)
                        # count for stats
                        count_E += 1
                        count_E_correct += int(E_dist.argmax() == next_E)
                        # calculate entity loss
                        E_loss += torch.nn.functional.cross_entropy(E_dist, next_E.view(-1))
                        e_div  += 1
                        
                        # register entity
                        model.register_predicted_entity(next_E)
                        
                        # set e_current to entity embedding e_t-1
                        e_current = model.get_entity_embedding(next_E)
                        
                        # predict length of entity and calculate loss
                        L_dist = model.get_next_L(h_t, e_current)
                        L_loss += torch.nn.functional.cross_entropy(L_dist, next_L.view(-1))
                        
                        l_div  += 1
                    else:
                        # only for stats and possibility to amplify loss
                        if R_dist.argmax():
                            # wrong True pred
                            doc_r_false_positive += 1
                            # extra loss
                            # R_loss += r_current_loss * rz_amplifier
                        else:
                            # correct False pred
                            # R_loss += r_current_loss
                            pass
                else:
                    # 2. Otherwise
                    # last L unequal 1, continuing entity mention
                    # set last new_L = last_L - 1
                    # new_R = last_R
                    # new_E = last_E
                    
                    # additional prediction for E to get more training cases
                    # (it also makes stats more comparable to deep-mind paper)
                    E_dist = model.get_next_E(h_t, t)
                    count_E += 1
                    count_E_correct += int(E_dist.argmax() == next_E)
                    E_loss += torch.nn.functional.cross_entropy(E_dist, next_E.view(-1))
                    e_div  += 1
                    pass
                
                # 3. Sample X, get distribution for next Token
                
                X_dist = model.get_next_X(h_t, e_current)
                X_loss += torch.nn.functional.cross_entropy(X_dist, next_X.view(-1))
                # 4. Advance the RNN on predicted token, here in training next token 
                h_t, states = model.forward_rnn(doc.X[t+1], states)
                h_t = h_t.squeeze(0)
                # new hidden state of next token from here (h_t, previous was actually h_t-1)
                
                # 5. Update entity state
                if next_R == 1:
                    model.update_entity_embedding(next_E, h_t, t)
                    # set e_current to embedding e_t
                    e_current = model.get_entity_embedding(next_E)
                    
                    
                # 6. Nothing toDo?
                
            ## End of Paper Algorithm

            r_true_positive  += doc_r_true_positive
            r_false_positive += doc_r_false_positive
            count_R          += doc_count_R
            doc_r_prec   = doc_r_true_positive / max((doc_r_true_positive+doc_r_false_positive), 1)
            doc_r_recall = doc_r_true_positive / max(doc_count_R, 1)
            doc_rf_score = 2*((doc_r_prec*doc_r_recall)/max(doc_r_prec+doc_r_recall, 1))

            R_loss = R_loss / max(doc_rf_score, 0.35)
            
            X_epoch_loss += X_loss.item()
            R_epoch_loss += R_loss.item()
            E_epoch_loss += E_loss.item()
            L_epoch_loss += L_loss.item()
            X_loss /= len(doc)
            R_loss /= max(r_div, 1)         
            E_loss /= max(e_div, 1)
            L_loss /= max(l_div, 1)
            
            epoch_tokens += len(doc)
            epoch_r_div  += r_div
            epoch_l_div += l_div
            epoch_e_div += e_div



            if optimizer:
                optimizer.zero_grad()
                loss = X_loss + R_loss + E_loss + L_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            if status_interval and i_doc % status_interval == 0:
                r_prec   = r_true_positive / max((r_true_positive+r_false_positive), 1)
                r_recall = r_true_positive / max(count_R, 1)
                rf_score  = 2*((r_prec*r_recall)/max(r_prec+r_recall, 1))
                print(f'Doc {i_doc}/{len(corpus)-1}: X_loss {X_epoch_loss / epoch_tokens:0.3}, R_loss {R_epoch_loss / epoch_r_div:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, L_loss {L_epoch_loss / epoch_l_div:0.3}, E_acc {count_E_correct/count_E:0.3}, R_prec {r_prec:0.3}, R_recall {r_recall:0.3}')
                sys.stdout.flush()
        
        seconds = round(time.time() - epoch_start)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        x_hour_and_ = f'{h} hours and '*bool(h)
        if optimizer:
            print(f'Epoch {epoch} finished after {x_hour_and_}{m} minutes.')
        else:
            print(f'Evaluation on "{corpus.partition}" partition finished after {x_hour_and_}{m} minutes.')
        r_prec   = r_true_positive / max((r_true_positive+r_false_positive), 1)
        r_recall = r_true_positive / max(count_R, 1)
        rf_score  = 2*((r_prec*r_recall)/max(r_prec+r_recall, 1))

        print(f'Loss: X_loss {X_epoch_loss / epoch_tokens:0.3}, R_loss {R_epoch_loss / epoch_r_div:0.3}, E_loss {E_epoch_loss / epoch_e_div:0.3}, L_loss {L_epoch_loss / epoch_l_div:0.3}, E_acc {count_E_correct/count_E:0.3}, R_prec {r_prec:0.3}, R_recall {r_recall:0.3}, R_Fscore {rf_score:0.3}')
        #print(f'GPU Mem: {round(torch.cuda.memory_allocated(0)/1024**3,1)}/{round(torch.cuda.max_memory_allocated(0)/1024**3,1)} GB, {round(torch.cuda.memory_cached(0)/1024**3,1)}/{round(torch.cuda.max_memory_cached(0)/1024**3,1)} GB')
        print()

        if optimizer:
            file_name = str_pattern.format(model.__class__.__name__, model.lstm.hidden_size, epoch)
            save_model(model, file_name)
            if eval_corpus:
                with torch.no_grad():
                    model.eval()
                    run_nlm(model, eval_corpus, status_interval=None, rz_amplifier=rz_amplifier)
                    model.train()


import torch, sys
from dataset import CorpusLoader
#from run_model import run_nlm, run_lme
from helpers import device, load_model
corpus = CorpusLoader(partition='train')
eval_corpus = CorpusLoader(partition='dev')

d = 64
model = EntityNLM(vocab_size=corpus.vocab_size, 
                        embedding_size=d, 
                        hidden_size=d,
                        dropout=0.15).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
str_pattern='{}_{}_epoch_{}.pklt'
run_nlm(model, corpus, optimizer, epochs=25, eval_corpus=eval_corpus, status_interval=250, rz_amplifier=1, str_pattern=str_pattern)
