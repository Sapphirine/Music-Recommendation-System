# we use tensorflow API to design and implement a matrix factorization model for predicting the song ratings on the song dataset.
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# load data
training_data = pd.read_csv("data/train.csv")
members_data = pd.read_csv("data/members.csv")
songs_data = pd.read_csv("data/songs.csv")
songs_extra = pd.read_csv("data/song_extra_info.csv")
dataset=training_data

# preprocess the song id and user id 
def check_cols(df, cols):
    return [(col, False) if len(dataset[col].unique())-1 == dataset[col].max() else (col, True) for col in cols]

def resone_gaps(df, col):
    adj_col_uni = df[col].sort_values().unique()
    adj_df = pd.DataFrame(adj_col_uni).reset_index().rename(columns = {0: col, 'index': "adj_%s"%(col,)})
    return pd.merge(adj_df, df, how="right", on=col)

from __future__ import print_function
index_cols = ["msno", "song_id"]
cols_check = check_cols(dataset, index_cols)
print_check = lambda check:print(*["%s needs fix!"%(c,) if f else "%s ok."%(c,) for c, f in check], sep="\n")
print("before fix:")
print_check(cols_check)
for col, needs_fix in cols_check:
    if needs_fix:
        dataset = resone_gaps(dataset, col)

print("\nafter fix")
print_check(check_cols(dataset, ["adj_msno", "adj_song_id"]))

# split the data to train and validation set
dataset = dataset.sample(frac=1, replace=False)
n_split = int(len(dataset)*.7)
trainset = dataset[:n_split]
validset = dataset[n_split:]

# build the model
def initialize_features(num_users, num_songs, dim):
  	user_features = tf.get_variable(
        "theta",
        shape = [num_users, dim],
        dtype = tf.float32,
        initializer = tf.truncated_normal_initializer(mean=0, stddev=.05)
    )
    song_features = tf.get_variable(
        "phi",
        shape = [num_songs, dim],
        dtype = tf.float32,
        initializer = tf.truncated_normal_initializer(mean=0, stddev=.05)
    )
    return user_features, song_features

def create_dataset(user_ids, song_ids, ratings):
    user_id_var = tf.constant(name="userid", value=user_ids)
    song_id_var = tf.constant(name="songid", value=song_ids)
    ratings_var = tf.constant(name="ratings", value=np.asarray(ratings, dtype=np.float32))
    return user_id_var, song_id_var, ratings_var
   
def lookup_features(user_features, song_features, user_ids, song_ids): 
    selected_user_features = tf.gather(user_features, user_ids)
    selected_song_features = tf.gather(song_features, song_ids)
    return selected_user_features, selected_song_features

def predict(selected_user_features, selected_song_features):
    selected_predictions = tf.reduce_sum(
        selected_user_features * selected_song_features,
        axis = 1
    )
    return selected_predictions

def mean_squared_difference(predictions, ratings):
    difference = tf.reduce_mean(tf.squared_difference(predictions, ratings))
    return difference

# set hyper parameters
emb_dim = 8
learning_rate = 50
epochs = 5000

# train model, define the tensorflow graph and create the session to compute the values.
with tf.Graph().as_default():
    with tf.variable_scope("features"):
        usr_embs, son_embs = initialize_features(len(dataset.adj_msno.unique()), len(dataset.adj_song_id.unique()), emb_dim)
    with tf.variable_scope("train_set"):
        train_data = trainset[["adj_msno", "adj_song_id", "target"]].values.T #shape(3, 700146)
        train_usr_ids, train_son_ids, train_ratings = create_dataset(*train_data)# expend to 3 lists
    with tf.variable_scope("valid_set"):
        valid_data = validset[["adj_msno", "adj_song_id", "target"]].values.T
        valid_usr_ids, valid_son_ids, valid_ratings = create_dataset(*valid_data)
    with tf.variable_scope("training"):
        train_sel_usr_emb, train_sel_son_emb = lookup_features(usr_embs, son_embs, train_usr_ids, train_son_ids)
        train_preds = predict(train_sel_usr_emb, train_sel_son_emb)
        train_loss = mean_squared_difference(train_preds, train_ratings)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_usr_embs = optimizer.minimize(train_loss, var_list=[usr_embs])
        train_son_embs = optimizer.minimize(train_loss, var_list=[son_embs])   
    with tf.variable_scope("validation"):
        valid_sel_usr_emb, valid_sel_son_emb = lookup_features(usr_embs, son_embs, valid_usr_ids, valid_son_ids)
        valid_preds = predict(valid_sel_usr_emb, valid_sel_son_emb)
        valid_loss = mean_squared_difference(valid_preds, valid_ratings)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('Graph/MF',sess.graph)
        sess.run(tf.global_variables_initializer())
        train_loss_history = []
        valid_loss_history = []
        for i in range(epochs):
            current_train_loss, _ = sess.run([train_loss, train_usr_embs])
            current_train_loss, _ = sess.run([train_loss, train_son_embs])
            current_valid_loss = sess.run(valid_loss)
            if i%100 ==0:
                print("valid loss at step %i: %f"%(i+1, current_valid_loss))
            train_loss_history.append(current_train_loss)
            valid_loss_history.append(current_valid_loss)
        final_user_features, final_song_features = sess.run([usr_embs, son_embs])
        final_valid_predictions = sess.run(valid_preds) 
        writer.close()

# plot the traing loss and valid loss
plt.figure(figsize=(12,10))
plt.plot(train_loss_history, color="red", label="train loss")
plt.plot(valid_loss_history, color="blue", label="valid loss")
plt.xlabel("epoch") 
plt.ylabel("loss")
plt.legend()
plt.show()

mf_accuracy = np.sum(np.round(final_valid_predictions) == validset.target.values) / float(len(final_valid_predictions))
print("MF Accuracy: %f%%"%(mf_accuracy*100,))

# result on validation set
results = validset.copy()
results["prediction (rnd.)"] = np.asarray(np.round(final_valid_predictions), dtype=np.int16)
results["prediction (prc.)"] = final_valid_predictions
results.head(50)

results = validset.copy()
results["prediction (rnd.)"] = np.asarray(np.round(final_valid_predictions), dtype=np.int16)
results["prediction (prc.)"] = final_valid_predictions
results.head(50)

results.loc[results["prediction (rnd.)"]>1, "prediction (rnd.)"] = 1
results.loc[results["prediction (rnd.)"]<0, "prediction (rnd.)"] = 0

# measures
def compute_recall(prediction_col, target_col):
    recall=[]
    for i in range(2):
        rating_df = results[results[target_col]==i]
        num_true_rating = len(rating_df)+0.0
        current_recall = (len(rating_df[rating_df[prediction_col]==i]))/num_true_rating
        recall.append(current_recall)
    return recall

def compute_precision(prediction_col, target_col): 
    precision=[]
    for i in range(2):
        pred_df = results[results[prediction_col]==i]
        pred_rating = len(pred_df)+0.0
        current_precision = (len(pred_df[pred_df[target_col]==i]))/pred_rating
        precision.append(current_precision)
    return precision    

def compute_mae(prediction_col, target_col):
    return np.mean(np.abs(results[prediction_col]-results[target_col]))

def compute_rmse(prediction_col, target_col):
    return np.sqrt(1.0/len(results)*np.sum((results[prediction_col]- results[target_col])**2))
    compute_recall('prediction (rnd.)', 'target')
    compute_precision('prediction (rnd.)', 'target')
    compute_mae('prediction (rnd.)', 'target')
    compute_rmse('prediction (rnd.)', 'target')