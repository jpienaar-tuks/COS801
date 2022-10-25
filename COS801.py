# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:10:39 2022

@author: Johann
"""
import os, re
import pandas as pd
import numpy as np
from time import time
from shutil import rmtree

from keras import Sequential, Model, Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.metrics import Precision, Recall
from tensorflow_addons.layers import CRF



from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import shuffle

EC_FILES = os.path.join(os.getcwd(),'labelled_contracts','elements_contracts')
EC_train_files = os.listdir(os.path.join(EC_FILES, 'train'))
EC_test_files = os.listdir(os.path.join(EC_FILES, 'test'))
SEQUENCE_WIDTH = 11

def zone_extractor_train(filename, target='STD', dataset='train', replace_linebreaks=None):
    r = re.compile(r'TOKEN_(\d+)\[([A-Z0]+)\]')
    with open(os.path.join(EC_FILES, dataset, filename),'rt') as f:
        tokens=[]
        for line in f.readlines():
            if replace_linebreaks!=None:
                line=line.replace('\n', replace_linebreaks)
            tokens.extend(r.findall(line))
    token_count=len(tokens)
    targets=[i[1] for i in tokens]
    pseudo_zones=[]
    pseudo_zone=[]
    for idx, token in enumerate(tokens):
        added=False
        if target in targets[max(0,idx-20):min(token_count,idx+20)]:
            pseudo_zone.append(token)
            added=True
        if not added and len(pseudo_zone)>0:
            pseudo_zones.append(pseudo_zone)
            pseudo_zone=[]
    if len(pseudo_zone)>0:
        pseudo_zones.append(pseudo_zone)
    
    final=[]
    for pseudo_zone in pseudo_zones:
        final.append([(int(i[0]), 1*(i[1]==target)) for i in pseudo_zone])
    return final

# def zone_extractor_test(filename, target='STD', dataset='test', replace_linebreaks=None):
#     r = re.compile(r'TOKEN_(\d+)\[([A-Z0]+)\]') #Token extractor
#     r2=re.compile(r'<(.+?)>(.*?)</(\1)>',re.DOTALL) # Zone extractor
#     sequences=[]
#     with open(os.path.join(EC_FILES, dataset, filename),'rt') as f:
#         f_text = ' '.join(f.readlines())
#         if replace_linebreaks!=None:
#             f_text = f_text.replace('\n', replace_linebreaks)
#     if target != 'LEG':
#         zones = r2.findall(f_text)
#         for zone in zones:
#             if zone[0] in zone_dict[target]:
#                 sequences.extend(r.findall(zone[1]))
#     else:
#         sequences = r.findall(f_text)
#     return [(int(i[0]), 1*(i[1]==target)) for i in sequences]



def prepare_sequence(input_sequence, sequence_width=11):
    """Generates X and y vectors for input into the model. Supposing that we want to learn the nouns in the sentence
    'The quick brown fox jumps over the lazy dog' 
    and we have an input sequence like:
    [('The',0), ('quick',0), ('brown',0), ('fox',1),('jumps', 0),('over', 0),('the',0),('lazy',0),('dog',1)]
    and a sequence_width = 3, prepares
    ['The', 'quick', 'brown'] - [0,0,0]
    ['quick', 'brown', 'fox'] - [0,0,1]
    ['brown','fox','jumps'] - [0,1,0]
    etc."""
    X=[]
    y=[]
    for i, v in enumerate(input_sequence[:-sequence_width+1]):
        X.append([t[0] for t in input_sequence[i:i+sequence_width]])
        y.append([t[1] for t in input_sequence[i:i+sequence_width]])
    return X, y

def make_bilstm(dropout=0.2):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embeddings], trainable=False))    
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=(SEQUENCE_WIDTH,embedding_dim)))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',Precision(), Recall()])
    model.summary()
    return model

def make_bilstm_lstm(dropout=0.2):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embeddings], trainable=False))    
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(300, return_sequences=True), input_shape=(SEQUENCE_WIDTH,embedding_dim)))
    model.add(LSTM(300, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',Precision(), Recall()])
    model.summary()
    return model

def make_bilstm_crf(dropout=0.2):                 #doesn't use the CRFModelWrapper
    inputs = Input(shape=(SEQUENCE_WIDTH,))
    model = Embedding(vocab_size, embedding_dim, weights=[embeddings], trainable=False)(inputs)
    model = Dropout(dropout)(model)
    model = Bidirectional(LSTM(300, return_sequences=True), input_shape=(SEQUENCE_WIDTH,embedding_dim))(model)
    model = Dropout(dropout)(model)
    crf = CRF(1)
    out = crf(model)[1]
    model=Model(inputs,out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',Precision(), Recall()])
    model.summary()
    return model

# def make_bilstm_crf(dropout=0.2):
#     inputs = Input(shape=(SEQUENCE_WIDTH,))
#     basemodel = Embedding(vocab_size, embedding_dim, weights=[embeddings], trainable=False)(inputs)
#     basemodel = Dropout(dropout)(basemodel)
#     basemodel = Bidirectional(LSTM(300, return_sequences=True), input_shape=(SEQUENCE_WIDTH,embedding_dim))(basemodel)
#     out = Dropout(dropout)(basemodel)
#     basemodel=Model(inputs,out)
#     model = CRFModelWrapper(basemodel, 1)
#     model.compile(optimizer='adam', metrics=['accuracy',Precision(), Recall()])
#     #model.build((SEQUENCE_WIDTH,))
#     #model.summary()
#     return model

def scoring(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true).flatten()
    y_pred = [1 if x > threshold else 0 for x in y_pred.flatten()]
    scores={'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred)}
    return scores

#Target to extraction zone lookup
zone_dict={'TIT':['COVER_PAGE','INTRODUCTION'],         #Contract Title 
           'CNP':['COVER_PAGE','INTRODUCTION'],         #Contract Party
           'STD':['COVER_PAGE','INTRODUCTION'],         #Start Date
           'EFD':['COVER_PAGE','INTRODUCTION'],         #Effective Date
           'TED':['TERMINATION','TERM'],                #Termination Date
           'PER':['TERM'],                              #Contract Period
           'VAL':['VALUE'],                             #Contract Value
           'GOV':['GOVERNING_LAW'],                     #Governing Law
           'JUR':['JURISDICTION'],                      #Jurisdiction
           'LEG': None,                                 #Legislation Refs.
           'HEAD': None}                                #Clause Headings

start = time()
# vocab_df=pd.read_csv(os.path.join(os.getcwd(),'dictionaries','encoded_vocabulary.csv'),sep=';')
# word_embedding_df=pd.read_csv(os.path.join(os.getcwd(),'dictionaries','word_embeddings.csv'), sep=';', 
#                               index_col=0, header=None)
# pos_tag_df=pd.read_csv(os.path.join(os.getcwd(),'dictionaries','pos_tag_embeddings.csv'), sep=';', header=None)
embeddings = pd.read_csv(os.path.join(os.getcwd(),'dictionaries','merged_embeddings.csv'), index_col=0, header=None).to_numpy()

word_embeddings = embeddings[:,:200]
POS_embeddings = embeddings[:,200:225]
token_shape_embeddings = embeddings[:,225:]

# Allows us to easily drop some embeddings for experimentation
embeddings = np.concatenate([word_embeddings, POS_embeddings, token_shape_embeddings], axis=1) # 


# embeddings = np.insert(embeddings,0,np.zeros((1,230)),0) # inserted a 0'th row in the csv, so this is no longer necessary
vocab_size, embedding_dim = embeddings.shape 
print('Reading embeddings took {:.2f} s'.format(time()-start))

# Tensorboard experiment - clear previous outputs
try:
    rmtree('logs')
except FileNotFoundError:
    pass

models={'BiLSTM': make_bilstm,
        'BiLSTM-LSTM': make_bilstm_lstm,
        'BiLSTM-CRF': make_bilstm_crf
        }

scores=[]
cb={} #Just meant to scheck class balance
for target in zone_dict.keys():
    if target in ['HEAD']:
        continue                    #TODO!!!
    train_zones=[]
    for f in EC_train_files:
        train_zones.extend(zone_extractor_train(f, target, 'train', 'TOKEN_0[0]'))
    
    test_zones=[]
    for f in EC_test_files:
        test_zones.extend(zone_extractor_train(f, target, 'test', 'TOKEN_0[0]'))
        
    X=[]
    y=[]
    for zone in train_zones:
        a, b = prepare_sequence(zone, SEQUENCE_WIDTH)
        X.extend(a)
        y.extend(b)
        
    X_train, X_val, y_train, y_val = train_test_split(*shuffle(X, y, random_state=0), test_size=0.2)
    
    X_test = []
    y_test = []
    for zone in test_zones:
        a, b = prepare_sequence(zone, SEQUENCE_WIDTH)
        X_test.extend(a)
        y_test.extend(b)
    
    cb[target]=(np.array(y_train).mean(axis=0), np.array(y_val).mean(axis=0),  np.array(y_test).mean(axis=0))
    
    for model_name, model_init in models.items():
        batch_size = 32
        if ('CRF' in model_name) and (target in ['STD', 'EFD', 'TED', 'VAL', 'LEG', 'GOV']):
            batch_size=64
            if target == 'LEG':
                batch_size = 128
        model = model_init(0.2)
        model.fit(X_train, y_train, epochs=2, batch_size=batch_size, validation_data=(X_val, y_val))
        model_score = scoring(y_test, model.predict(X_test))
        for metric, value in model_score.items():
            scores.append([target, model_name, metric, value])
duration = time()-start
print('That took a grand total of {}s, or {:.2f} min'.format(duration, duration/60))
pd.DataFrame(scores, columns=['Target','Model','Metric','Value']).to_csv('scores.csv')
        

    
       