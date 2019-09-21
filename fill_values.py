
import sys
import numpy as np
import pandas as pd
# import joblib just to save our model
from sklearn.externals import joblib

# import function that we implemented in relevant python source files
from train_model import transform_data



if __name__ == '__main__':

	model_name = sys.argv[1]
	model = joblib.load(model_name)

	# read train/val/test set from csv files
	df_train_origin = pd.read_csv("train_set.csv", sep=',', encoding='latin-1', dtype=str)
	df_val_origin   = pd.read_csv("val_set.csv", sep=',', encoding='latin-1', dtype=str)
	df_test_origin  = pd.read_csv("test_set.csv", sep=',', encoding='latin-1', dtype=str)
	
	df_train = df_train_origin.drop(columns='session_id')
	df_val   = df_val_origin.drop(columns='session_id')
	df_test  = df_test_origin.drop(columns='session_id')
	print("Shape of training set: {} --- Shape of validation set: {} --- Shape of test set: {}".format(df_train.shape, df_val.shape, df_test.shape))

	# use numpy matrices instead of pandas dataframes
	X_train = df_train.as_matrix().astype(float)
	X_val   = df_val.as_matrix().astype(float)
	X_test  = df_test.as_matrix().astype(float)
	
	feat_num = X_train.shape[1]
	X_train = np.reshape(X_train, (X_train.shape[0], 1, feat_num))
	X_val   = np.reshape(X_val, (X_val.shape[0], 1, feat_num))
	X_test   = np.reshape(X_test, (X_test.shape[0], 1, feat_num))
	
	X_train = np.vectorize(transform_data)(X_train)
	X_val   = np.vectorize(transform_data)(X_val)
	X_test  = np.vectorize(transform_data)(X_test)
	print("Shape of training set: {} --- Shape of validation set: {}".format(X_train.shape, X_val.shape))
	
	result_train = model.predict(X_train)
	result_val	 = model.predict(X_val)
	result_test  = model.predict(X_test)

	X_train = np.reshape(X_train, (X_train.shape[0], feat_num))
	X_val   = np.reshape(X_val, (X_val.shape[0], feat_num))
	X_test   = np.reshape(X_test, (X_test.shape[0], feat_num))

	result_train = np.vectorize(lambda x,y : x and y)(X_train, result_train)
	result_val	 = np.vectorize(lambda x,y : x and y)(X_val, result_val)
	result_test  = np.vectorize(lambda x,y : x and y)(X_test, result_test)


	for i in range(len(df_train)):
		df_train.iloc[i, :] = result_train[i,:]

	for i in range(len(df_val)):
		df_val.iloc[i, :] = result_val[i,:]

	for i in range(len(df_test)):
		df_test.iloc[i, :] = result_test[i,:]

	df_train = pd.concat([df_train_origin['session_id'], df_train], axis=1)
	df_val 	 = pd.concat([df_val_origin['session_id'], df_val], axis=1)
	df_test  = pd.concat([df_test_origin['session_id'], df_test], axis=1)

	with open("train_filled.csv", "w") as ftrain, open("val_filled.csv", "w") as fval, open("test_filled.csv", "w") as ftest:
		ftrain.write(df_train.to_csv(index=False))
		fval.write(df_val.to_csv(index=False))
		ftest.write(df_test.to_csv(index=False))
	