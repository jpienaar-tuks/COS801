import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

print('Reading and preprocessing csv\'s')
dictionary_dir = os.path.join(os.getcwd(),'dictionaries')
encoded_vocab_df = pd.read_csv(os.path.join(dictionary_dir,'encoded_vocabulary.csv'), sep=';')
token_count = encoded_vocab_df.shape[0]
general_df = encoded_vocab_df.loc[:,['GENERAL' in i for i in encoded_vocab_df.columns]]
pca_5 = PCA(n_components=5)
token_shape_df = pca_5.fit_transform(general_df)

pos_tag_df = pd.read_csv(os.path.join(dictionary_dir,'pos_tag_embeddings.csv'), sep=';', header=None)
word_embed_df = pd.read_csv(os.path.join(dictionary_dir,'word_embeddings.csv'),sep=';', header=None)

print('Start main loop...')
merged_embeddings = np.random.uniform(-3,3, size=(token_count, 231)) #1 index, 200 word embedding + 25 pos embedding + 5 token shape embedding
for i, row in encoded_vocab_df.iterrows():
    if i%1000==0:
        print('Processing row {} of {}...'.format(i, token_count))
    merged_embeddings[i,0] = int(row['TOKEN_CODE'].split('_')[1])
    wec = row['WORD_EMBEDDING_CODE']
    if wec =='UNK':
        #merged_embeddings[i,1:201] = word_embed_df.mean(axis=0)[1:]
        #merged_embeddings[i,1:201] = np.random.uniform(-3,3,200)
        pass # Just leave the random initialisations
    else:
        merged_embeddings[i,1:201] = word_embed_df.iloc[int(wec),1:]
    merged_embeddings[i,201:226] = pos_tag_df.loc[pos_tag_df[0]==row['POS_TAG'],1:]
    merged_embeddings[i, 226:] = token_shape_df[i,:]
merged_embeddings = np.insert(merged_embeddings,0,np.zeros((1,231)),0) # Insert a 0'th row with all zero vector
df=pd.DataFrame(merged_embeddings)
df[0] = df[0].astype('int32')
df.to_csv('merged_embeddings.csv', header=False, index=False)



