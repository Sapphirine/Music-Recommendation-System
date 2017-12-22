
# coding: utf-8

# Here we use the tf.esimator API to train a wide linear model and a deep feed-forward neural network, which combines the strengths of memorization and generalization.
# 
# The explaination of the model from tensorflow website as following:
# 
# At a high level, here are the steps using the tf.estimator API:
# 
# 1. Preprocess our song-user dataset in pandas.
# 2. Define features
# 3. Build inputs from the original dataset 
# 4. Hash string type categorical features and use int type features value as category id directly.
# 5. Create embeddings of sparse features for the deep model.
# 6. Define features for both the deep and the wide part of the model.
# 7. Train and validate the model.

# In[1]:

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')


# ### 1. data preprocessing

# load dataset and split train and valid set.

# In[2]:

dataset = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")


# In[3]:

member_data = pd.read_csv('members.csv')
member_data['member_Index'] = list(range(len(member_data)))
song_data = pd.read_csv('songs.csv')
song_extra = pd.read_csv('song_extra_info.csv')
song_data = pd.merge(song_data, song_extra, how = 'left', on = "song_id")
song_data['song_Index'] = list(range(len(song_data)))
dataset = pd.merge(pd.merge(dataset, member_data, how="left", on="msno"), song_data, how = 'left', on = 'song_id')
dataset


# # process feature 

# In[ ]:

dataset.info()


#  <b>age</b>

# In[4]:

dataset = dataset[dataset.bd > 0]
dataset = dataset[dataset.bd < 100]


# <b> gender</b>

# In[5]:

# dataset.loc[dataset.gender.isnull(), 'gender' ]='unknown'
dataset = dataset[dataset.gender.notnull()]


# In[6]:

dataset = dataset[dataset.song_length.notnull()]
dataset = dataset[dataset.language.notnull()]
dataset = dataset[dataset.genre_ids.notnull()] 
dataset.loc[dataset.composer.isnull(), 'composer'] = 'unknown'
dataset.loc[dataset.lyricist.isnull(), 'lyricist'] = 'unknown'
dataset.loc[dataset.name.isnull(), 'name'] = 'unknown'
dataset.loc[dataset.isrc.isnull(), 'isrc'] = 'unknown'
dataset = dataset[dataset['source_screen_name'].notnull()]
dataset = dataset[dataset['source_type'].notnull()]
dataset = dataset[dataset['source_system_tab'].notnull()]


# In[7]:

dataset.msno = dataset.member_Index
dataset.song_id = dataset.song_Index
adj_col = dataset['song_id']
adj_col_uni = adj_col.sort_values().unique()


# In[8]:

adj_df = pd.DataFrame(adj_col_uni).reset_index().rename(columns = {0:'song_id','index':'adj_song_id'})
dataset = pd.merge(adj_df,dataset,how="right", on="song_id")
dataset['adj_msno'] = dataset['msno']


# ### 2. define features

# By looking at the dataset, we know that we have following features: "genre_ids", "zipcode", "gender","source_type"， "bd", "song_length","registered_via".
#  - The data type are string or int, so we group the features in STR and INT groups accordingly for further encoding.
#  - We select all features for the deep model
#  - We select some feature transformation for the wide model.<br>

# In[15]:

dataset.info()


# In[28]:

dataset = dataset[dataset.source_system_tab!='null']


# In[29]:

dataset.shape


# In[12]:

dataset.isnull().sum()


# In[ ]:

# dataset.language = dataset.language.astype(int)
# dataset.song_length = dataset.song_length.astype(int)


# In[ ]:

sorted(dataset.genre_ids.unique().tolist())


# In[13]:

def chage_genre(genre):
    if "|" in genre:
        genre = genre.partition('|')[0]
    return genre


# In[14]:

dataset.genre_ids = dataset.genre_ids.apply(chage_genre)


# In[40]:

def split_dataset(dataset, split_frac=.8):
    dataset = dataset.sample(frac=1, replace=False)
    n_split = int(len(dataset)*split_frac)
    trainset = dataset[:n_split]
    validset = dataset[n_split:]
    return trainset, validset

fullset = dataset
trainset, validset = split_dataset(fullset)


# In[ ]:

dataset.song_length.describe()
dataset.bd.describe()
dataset.registered_via.unique()


# In[55]:

CAT_STR_COLS = ["gender"]#,"source_type""gender",
CAT_INT_COLS = ["city"]#,"song_length","genre_ids","registered_via"
LABEL_COL = "target"
DEEP_COLS = CAT_STR_COLS + CAT_INT_COLS
WIDE_COL_CROSSES = [["gender", "city"]]#, ["gender", "registered_via"]


# In[51]:

dataset.city.head()


# In[52]:

dataset.bd.head()


# ### 3. build inputs from original dataset

# Since this Deep and Widel Model API expects sparse tensors as inputs, we convert here all the feature columns and the label column from our original dataset to sparse tensors.

# In[42]:

def make_inputs(dataframe):
    feature_inputs = {
        col_name: tf.SparseTensor(
            indices = [[i, 0] for i in range(len(dataframe[col_name]))],
            values = dataframe[col_name].values,
            dense_shape = [len(dataframe[col_name]), 1]
        )
        for col_name in CAT_STR_COLS + CAT_INT_COLS
    }
    label_input = tf.constant(dataframe[LABEL_COL].values-1)
    return (feature_inputs, label_input)


# ### 4. create hash buckets for categorical features

# Here we define two functions to encode string type categorical features and int type categorical features.

# In[43]:

def make_hash_columns(CAT_STR_COLS):

    
    hashed_columns = [
        tf.feature_column.categorical_column_with_hash_bucket(col_name, hash_bucket_size=1000) 
        for col_name in CAT_STR_COLS
    ]
    return hashed_columns


# In[44]:

def make_int_columns(CAT_INT_COLS):   
    int_columns = [
        tf.feature_column.categorical_column_with_identity(col_name, num_buckets=1000, default_value=0)
        for col_name in CAT_INT_COLS
    ]
    return int_columns


# ### 5. create embedding for sparse features 

# Create dense embeddings for the sparse features to feed into DNN.

# In[45]:

def make_embeddings(hashed_columns, int_columns, dim=6):
    embedding_layers = [
        tf.feature_column.embedding_column(
            column,
            dimension=dim
        )
        for column in hashed_columns+int_columns
    ]
    return embedding_layers


# ### 6. define features for the wide part

#  all the columns from embedding layers should go into the deep model, so we our deep model input equals to embedding_layers and we are not going to write a function for this.

# In[46]:

def make_wide_input_layers(WIDE_COL_CROSSES):
    crossed_wide_input_layers = [
        tf.feature_column.crossed_column([c for c in cs], hash_bucket_size=int(10**(3+len(cs))))
        for cs in WIDE_COL_CROSSES
    ]
    return crossed_wide_input_layers


# ### 7. train and validate the model

# Here we provide input features for the deep model and wide model, define the number of layers and layer sizes of DNN 
# and create the model with tf.contrib.learn.DNNLinearCombinedClassifier. We save the model in directory ./model/

# In[56]:

from __future__ import print_function
print("create input layers...", end="")
hash_columns = make_hash_columns(CAT_STR_COLS)
int_columns = make_int_columns(CAT_INT_COLS)
embedding_layers = make_embeddings(hash_columns, int_columns,dim =6)
deep_input_layers = embedding_layers
wide_input_layers = make_wide_input_layers(WIDE_COL_CROSSES)
print("done!")
print("create model...", end="")
model = tf.contrib.learn.DNNLinearCombinedClassifier(
    n_classes=2,
    linear_feature_columns = wide_input_layers,
    dnn_feature_columns = deep_input_layers,
    dnn_hidden_units = [32, 16],
    fix_global_step_increment_bug=True,
    config = tf.contrib.learn.RunConfig(
        keep_checkpoint_max = 1,
        save_summary_steps = 10,
        model_dir = "./model/"
    )
)
print("done!")
print("training model...", end="")
model.fit(input_fn = lambda: make_inputs(trainset), steps=1000)
print("done!")
print("evaluating model...", end="")
results = model.evaluate(input_fn = lambda: make_inputs(validset), steps=1)
print("done!")
print("calculating predictions...", end="")
predictions = model.predict_classes(input_fn = lambda: make_inputs(validset))
print("done!")
print("calculating probabilites...", end="")
probabilities = model.predict_proba(input_fn = lambda: make_inputs(validset))
print("done!")


# In[ ]:

for n, r in results.items():
    print("%s: %a"%(n, r))


# 

# In[ ]:

predict = list(predictions)


# In[ ]:

prob = list(probabilities)


# In[ ]:

dnw_accuracy = np.sum(np.asarray(predict)+1 == validset.target.values) / float(len(validset))
print("DNW Accuracy: %f%%"%(dnw_accuracy*100,))


# In[ ]:


results = validset.copy()
results["prediction"] = np.asarray(predict)+1
results["rating1"] = np.vstack(prob)[:,0]
results["rating2"] = np.vstack(prob)[:,1]

results.head(5)


# In[ ]:



