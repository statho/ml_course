import sys
import numpy as np
import pandas as pd
import scipy
from train_model import transform_data
from sklearn.externals import joblib


def create_dict(dfr):
	# create a dictionary to save for each inital feature, its corresponding indices after the one-hot encoding
	prev = ''
	feat_dict = {}
	for i, key in enumerate(dfr.keys()):
		curr = key.split("__")[0]
		if curr != prev:
			feat_dict[curr] = [i]
		else:
			feat_dict[curr].append(i)
		prev = curr
	return feat_dict


df_train_origin = pd.read_csv("train_set.csv", sep=',', encoding='latin-1', dtype=str)
df_val_origin = pd.read_csv("val_set.csv", sep=',', encoding='latin-1', dtype=str)
df_test_origin = pd.read_csv("test_set.csv", sep=',', encoding='latin-1', dtype=str)

df_train = df_train_origin.drop(columns='session_id')
df_val = df_val_origin.drop(columns='session_id')
df_test = df_test_origin.drop(columns='session_id')

df_total = df_train.append(df_val)
df_total = df_total.append(df_test)

X_total = df_total.values
feat_dict = create_dict(df_total)

# generate random data points
n_rand_feats = 10
n_generated_samples = 100
#initialize generated data
X_generated = np.zeros((n_generated_samples, X_total.shape[1]))

#fill the generated data
for i in range(n_generated_samples):
    # select features to fill at random
    rand_feats = np.random.randint(0, len(list(feat_dict.keys())), n_rand_feats)
    for j in range(n_rand_feats):
        X_generated[i,np.random.choice(feat_dict[list(feat_dict.keys())[rand_feats[j]]])] = 1


#predict the missing features

model_name = sys.argv[1]
model = joblib.load(model_name)

feat_num = X_generated.shape[1]
X_generated = np.reshape(X_generated, (X_generated.shape[0], 1, feat_num))
X_generated = np.vectorize(transform_data)(X_generated)
generated_data =  model.predict(X_generated)

# sample 100 data points from the original data set at random

idcs = np.random.choice(range(X_total.shape[0]), n_generated_samples)
real_data = X_total[:, idcs]


# sample 1000 data points from the original data for comparison
idcs = np.random.choice(set(range(X_total.shape[0]))-set(idcs), 1000)
comp_data = X_total[:, idcs]

distances_real = scipy.spatial.distance.cdist(real_data, comp_data)
distances_gen = scipy.spatial.distance.cdist(X_generated, comp_data)
