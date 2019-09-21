
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def count_empty_cells(col):
	return len(list(filter(lambda x: x in ['', ' ', 'nan', 'NaN', np.nan], col)))

def count_missing_per_feature(df):
	to_pred_dict = {}
	for key in df.keys():
		c = count_empty_cells(df[key])
		if c > 0:
			to_pred_dict[key] = c
	return to_pred_dict

def count_missing_per_sample(df):
	missing = []
	for _, row in df.iterrows():
		missing.append(count_empty_cells(row))
	return missing

def complete(dest, source):
	for i in np.intersect1d(np.where(df[dest] == ''),np.where(df[source] != '')):
		print(df[dest][i],df[source][i])
		df[dest][i] = df[source][i]
	print("LOL")


def date2season(month):
	if month in ['1', '2', '12']:
		return "winter"
	if month in ['3', '4', '5']:
		return "spring"
	if month in ['6', '7', '8']:
		return "summer"
	return "fall"

def race_replace(race):
	if race == "chinese":
		return str(2)
	elif race == "malay":
		return str(3)
	elif race == "brazilwhite" or race == "dutch":
		return str(6)
	elif race == "brazilblack":
		return str(5)
	elif race == "brazilbrown":
		return str(7)
	return race

def art_preferance(x):
	if x == '':
		return x
	x = float(x)
	if x > 0:
		return '1'
	return '0'


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


if __name__ == '__main__':


	filename = sys.argv[1]
	#filename = "../ML1/Tab.delimited.Cleaned.dataset.WITH.variable.labels.csv"
	
	# read csv file and drop garbage columns
	df = pd.read_csv(filename, sep='\t', encoding='latin-1', dtype=str)
	print("Inital shape of dataframe: {}".format(df.shape))
	
	dv_list   = ['sunkDV', 'gainlossDV', 'anchoring1', 'anchoring2', 'anchoring3', 'anchoring4', 'Ranch1', 'Ranch2', 'Ranch3', 'Ranch4', 'scales',\
				 'reciprocityother', 'reciprocityus', 'allowedforbidden', 'quote', 'flagdv', 'Sysjust', 'Imagineddv', 'IATexpart', 'IATexpmath', 'IATexp.overall']

	drop_list = ['user_id', 'last_update_date', 'session_last_update_date', 'creation_date', 'session_creation_date',\
				 'expcomments', 'numparticipants_actual', 'numparticipants', 'sample', 'beginlocaltime', 'text', 'session_status',\
				 'previous_session_id', 'feedback', 'previous_session_schema', 'user_agent', 'task_status', 'task_sequence',\
				 'session_created_by', 'study_url', 'sunkgroup', 'gainlossgroup', 'anch1group', 'anch2group', 'anch3group', \
				 'anch4group', 'gambfalgroup', 'gambfalDV', 'gamblerfallacya_sd', 'gamblerfallacyb_sd', 'scalesgroup',\
				 'reciprocitygroup', 'allowedforbiddenGroup', 'quoteGroup', 'flagGroup', 'MoneyGroup', 'ContactGroup', 'study_name',\
				 'Ranchori', 'RAN001', 'RAN002', 'RAN003', 'd_donotuse', 'iatorder', 'exprunafter2', 'scalesreca', 'scalesrecb',\
				 'quotearec', 'quotebrec', 'flagtimeestimate1', 'flagtimeestimate2', 'flagtimeestimate3', 'flagtimeestimate4',\
				 'noflagtimeestimate1', 'noflagtimeestimate2', 'noflagtimeestimate3', 'noflagtimeestimate4', 'totalflagestimations',\
				 'totalnoflagtimeestimations', 'moneyagea', 'moneyageb', 'moneyethnicitya', 'moneyethnicityb', 'moneygendera', 'moneygenderb',\
				 'partgender', 'imagineddescribe', 'IATfilter', 'totexpmissed', 'IATEXPfilter', 'citizenship', 'imptaskto',\
				 'nativelang', 'nativelang2', 'citizenship2', 'omdimc3rt', 'omdimc3trt', 'anchoring1akm', 'anchoring1bkm','iat_exclude',\
				 'anchoring3ameter', 'anchoring3bmeter', 'religion', 'filter_$', 'race', 'mturk.non.US', 'mturk.Submitted.PaymentReq',
				 'mturk.total.mini.exps', 'mturk.duplicate', 'mturk.exclude.null', 'mturk.keep', 'mturk.exclude', 'meanlatency', 'meanerror', \
				 'block2_meanerror', 'block3_meanerror', 'block5_meanerror', 'block6_meanerror', 'lat11', 'lat12', 'lat21', 'lat22', 'sd1', 'sd2', 'd_art1', 'd_art2'] \
				  + ['o'+str(i) for i in range(1,12)] + ['task_id.'+str(i) for i in range(46)] + ['task_url.'+str(i) for i in range(46)]\
				  + ['task_creation_date.'+str(i) for i in range(46)] + ['priorexposure'+str(i) for i in range(1,14)]
	
	drop_list = drop_list + dv_list
	df = df.drop(columns=drop_list)
	print("Shape of dataframe after some features dropped: {}".format(df.shape))
	
	metadata_list = ['session_date', 'referrer', 'expgender', 'exprace', 'runalone', 'compensation', 'recruitment', 'separatedornot', 'age',\
					 'flag-american', 'money-first', 'ethnicity', 'major', 'omdimc3-pass', 'politicalid', 'sex', 'scalesorder',\
					 'reciprocorder', 'diseaseforder','quoteorder', 'flagprimorder', 'sunkcostorder', 'anchorinorder', 'allowedforder', 'gamblerforder',\
					 'moneypriorder', 'imaginedorder']


	# replace '.' with whitespace and and remove all leading and trailing whitespaces form strings
	df['sex']   	= df['sex'].replace({'f':'female','m': 'male', '.': 'prefernot'})
	df['expgender'] = df['expgender'].replace({'.': 'prefernot'})
	df['ethnicity'] = df['ethnicity'].replace({'.': '3'})
	df = df.apply(lambda x: x.replace('.',' ') if x.dtype == "object" else x)
	df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
	

	## -----
	# preprocess features to use
	
	## change the values of some features to make them more useful
	#

	# metadata -- don't predict
	df['exprace']      = df['exprace'].apply(lambda x: race_replace(x))
	df['session_date'] = df['session_date'].apply(lambda x: date2season( x.split("/")[0] ) )
	
	df['session_date'] = df['session_date'].replace({'summer': '0', 'fall': '1'})
	df['exprace'] = df['exprace'].replace({'1':'American-Indian/Alaska-Native','2':'East-Asian','3':'South-Asian',\
										   '4':'Native-Hawaiian/Pacific-Islander','5':'Black/African-American','6':'White',\
										   '7':'More-than-one-race-Black/White','8':'More-than-one-race-Other','9':'Other/Unknown',\
										   '10':'Hispano/Latino'})
	df['exprunafter'] = df['exprunafter'].replace({'runafter': '0', 'runalone' : '1'})
	
	#age_classes = [12:18, 19:22, 23:29, 30:49, 50:90]
	age_buckets = [0, 11, 18, 22, 29, 49, 100]
	df['age']   = df['age'].replace({'': '1'})
	df['age']   = pd.to_numeric(df['age'], errors='ignore')
	df['age']   = pd.cut(df['age'], age_buckets, labels=["", "12-18", "19-22", "23-29", "30-49", "50-100"])
	
	df['order'] = df['order'].replace({'1': '0', '2': '1'})
	df['recruitment']   = df['recruitment'].replace({'othersubjpool': 'other', 'advertisements': 'other'})
	df['lab_or_online'] = df['lab_or_online'].replace({'In-lab': '0', 'Online': '1'})
	df['flagfilter']    = df['flagfilter'].replace({'exclude': '0', 'include': '1'})
	df['omdimc3']       = df['omdimc3'].replace({'Fail': '0', 'Pass': '1'})
	df['us_or_international'] = df['us_or_international'].replace({'US': '0', 'International' : '1'})

	# questions -- features to predict
	df['allowedforbiddena'] = df['allowedforbiddena'].replace({'No': '0', 'Yes': '1'})
	df['allowedforbiddenb'] = df['allowedforbiddenb'].replace({'No': '0', 'Yes': '1'})
	df['diseaseframinga']   = df['diseaseframinga'].replace({'200 people will be saved': '0', '1/3 probability to save all, 2/3 nobody will be saved' : '1'})
	df['diseaseframingb']   = df['diseaseframingb'].replace({'400 people will die': '0', '1/3 probability nobody will die, 2/3 that 600 will die' : '1'})
	df['flagsupplement1']   = df['flagsupplement1'].replace({'Not at all': '1', 'Very much': '11'})
	df['flagsupplement2']   = df['flagsupplement2'].replace({'Democrat': '1', 'Republican': '7'})
	df['flagsupplement3']   = df['flagsupplement3'].replace({'Liberal': '1', 'Conservative': '7'})
	
	df['iatexplicitart1']   = df['iatexplicitart1'].replace({'Moderately bad' : '6', 'Very bad': '7'})
	df['iatexplicitart2']   = df['iatexplicitart2'].replace({'Moderately Sad' : '6', 'Very Sad': '7'})
	df['iatexplicitart3']   = df['iatexplicitart3'].replace({'Moderately Ugly' : '6', 'Very Ugly': '7'})
	df['iatexplicitart4']   = df['iatexplicitart4'].replace({'Moderately Disgusting' : '6', 'Very Disgusting': '7'})
	df['iatexplicitart5']   = df['iatexplicitart5'].replace({'Moderately Avoid' : '6', 'Very Avoid': '7'})
	df['iatexplicitart6']   = df['iatexplicitart6'].replace({'Moderately Afraid' : '6', 'Very Afraid': '7'})
	
	df['iatexplicitmath1']  = df['iatexplicitmath1'].replace({'Slightly bad': '5', 'Moderately bad' : '6', 'Very bad': '7'})
	df['iatexplicitmath2']  = df['iatexplicitmath2'].replace({'Slightly Sad': '5', 'Moderately Sad' : '6', 'Very Sad': '7'})
	df['iatexplicitmath3']  = df['iatexplicitmath3'].replace({'Slightly Ugly': '5', 'Moderately Ugly' : '6', 'Very Ugly': '7'})
	df['iatexplicitmath4']  = df['iatexplicitmath4'].replace({'Slightly Disgusting': '5', 'Moderately Disgusting' : '6', 'Very Disgusting': '7'})
	df['iatexplicitmath5']  = df['iatexplicitmath5'].replace({'Slightly Avoid': '5', 'Moderately Avoid' : '6', 'Very Avoid': '7'})
	df['iatexplicitmath6']  = df['iatexplicitmath6'].replace({'Slightly Afraid': '5', 'Moderately Afraid' : '6', 'Very Afraid': '7'})

	df['reciprocityusa']    = df['reciprocityusa'].replace({'No': '0', 'Yes': '1'})
	df['reciprocityusb']    = df['reciprocityusb'].replace({'No': '0', 'Yes': '1'})
	df['reciprocityothera'] = df['reciprocityothera'].replace({'No': '0', 'Yes': '1'}) 
	df['reciprocityotherb'] = df['reciprocityotherb'].replace({'No': '0', 'Yes': '1'})
	
	for i in range(1,9):
		df['sysjust'+str(i)] = df['sysjust'+str(i)].replace({'Strongly disagree': '1', 'Strongly agree': '7'})


	df['d_art']    = df['d_art'].apply(lambda x: art_preferance(x))
	df['artwarm']  = df['artwarm'].replace({'': '-1'})
	df['artwarm']  = pd.to_numeric(df['artwarm'], errors='ignore')
	df['artwarm']  = pd.cut(df['artwarm'], [-2, -1, 25, 50, 75, 100], labels=['', '0-25', '26-50', '51-75', '76-100'])
	df['mathwarm'] = df['mathwarm'].replace({'': '-1'})
	df['mathwarm'] = pd.to_numeric(df['mathwarm'], errors='ignore')
	df['mathwarm'] = pd.cut(df['mathwarm'], [-2, -1, 25, 50, 75, 100], labels=['', '0-25', '26-50', '51-75', '76-100'])
	
	df['gamblerfallacya'] = df['gamblerfallacya'].replace({'': '-1'})
	df['gamblerfallacya'] = pd.to_numeric(df['gamblerfallacya'], errors='ignore')
	df['gamblerfallacya'] = pd.cut(df['gamblerfallacya'], [-2, -1, 0, 1, 2, 3, 4, 100], labels=['', '0', '1', '2', '3', '4', '5-and-more'])
	df['gamblerfallacyb'] = df['gamblerfallacyb'].replace({'': '-1'})
	df['gamblerfallacyb'] = pd.to_numeric(df['gamblerfallacyb'], errors='ignore')
	df['gamblerfallacyb'] = pd.cut(df['gamblerfallacyb'], [-2, -1, 0, 1, 2, 3, 4, 100], labels=['', '0', '1', '2', '3', '4', '5-and-more'])
	
	# --- change variables from continuous to discrete ---	
	# change anchoring1a variables from numerical to classes, with 500 per class. minimum is 1501, maximum is 5903.015
	anchoring1a_bins = range(1500, 6001, 500) # 9 bins
	df['anchoring1a'] = pd.to_numeric(df['anchoring1a'], errors='ignore')
	df['anchoring1a'] = pd.cut(df['anchoring1a'], anchoring1a_bins)

	# change anchoring1b variables from numerical to classes, with 500 per class. minimum is 1553, maximum is 5999
	anchoring1b_bins = range(1500, 6001, 500) # 9 bins
	df['anchoring1b'] = pd.to_numeric(df['anchoring1b'], errors='ignore')
	df['anchoring1b'] = pd.cut(df['anchoring1b'], anchoring1b_bins)

	# change anchoring2a variables from numerical to classes with 400000 in each bin min is 200001, max is 4521987
	anchoring2a_bins = range(200000, 4600001, 400000) # 12 bins
	df['anchoring2a'] = pd.to_numeric(df['anchoring2a'], errors='ignore')
	df['anchoring2a'] = pd.cut(df['anchoring2a'], anchoring2a_bins)

	# change anchoring2b variables from numerical to classes with 400000 in each bin min is 236785, max is 4999999
	anchoring2b_bins = range(200000, 5000001, 400000)  # 13 bins
	df['anchoring2b'] = pd.to_numeric(df['anchoring2b'], errors='ignore')
	df['anchoring2b'] = pd.cut(df['anchoring2b'], anchoring2b_bins)

	# change anchoring3a variables from numerical to classes with 4300 in each bin min is 2001, max is 45000
	anchoring3a_bins = range(2000, 45001, 4300)  # 13 bins
	df['anchoring3a'] = pd.to_numeric(df['anchoring3a'], errors='ignore')
	df['anchoring3a'] = pd.cut(df['anchoring3a'], anchoring3a_bins)

	# change anchoring3b variables from numerical to classes with 4360 in each bin min is 2432, max is 45499
	anchoring3b_bins = range(2400, 46001, 4360)  # 11 bins
	df['anchoring3b'] = pd.to_numeric(df['anchoring3b'], errors='ignore')
	df['anchoring3b'] = pd.cut(df['anchoring3b'], anchoring3b_bins)

	# change anchoring4a variables from numerical to classes with 4790 in each bin min is 101, max is 48000
	anchoring4a_bins = range(100, 48001, 4790)  # 11 bins
	df['anchoring4a'] = pd.to_numeric(df['anchoring4a'], errors='ignore')
	df['anchoring4a'] = pd.cut(df['anchoring4a'], anchoring4a_bins)

	# change anchoring4b variables from numerical to classes with 4988 in each bin min is 120, max is 49999
	anchoring4b_bins = range(120, 50000, 4988)  # 11 bins
	df['anchoring4b'] = pd.to_numeric(df['anchoring4b'], errors='ignore')
	df['anchoring4b'] = pd.cut(df['anchoring4b'], anchoring4b_bins)


	# rename columns in the dataset
	df = df.rename(index=str, \
				  columns={'exprunafter': 'runalone', 'lab_or_online': 'exp-online', 'us_or_international': 'subject-international',\
				  		   'allowedforbiddena': 'forbidden', 'allowedforbiddenb': 'allowed', 'diseaseframinga':'disease-save-choseprob', \
				  		   'diseaseframingb':'disease-kill-choseprob', 'flagfilter': 'flag-american', 'moneyfilter': 'money-first', \
				  		   'flagsupplement1': 'flagsuppl-american', 'flagsupplement2': 'flagsuppl-republican', 'flagsupplement3': 'flagsuppl-conservative',\
				  		   'iatexplicitart1': 'art-good2bad', 'iatexplicitart2': 'art-happy2sad', 'iatexplicitart3': 'art-beautiful2ugly', 'iatexplicitart4': 'art-delightful2disgusting',\
				  		   'iatexplicitart5': 'art-approach2avoid', 'iatexplicitart6': 'art-unafraid2afraid', 'omdimc3': 'omdimc3-pass',\
				  		   'iatexplicitmath1': 'math-good2bad', 'iatexplicitmath2': 'math-happy2sad', 'iatexplicitmath3': 'math-beautiful2ugly', 'iatexplicitmath4': 'math-delightful2disgusting',\
				  		   'iatexplicitmath5': 'math-approach2avoid', 'iatexplicitmath6': 'math-unafraid2afraid', 'quotea': 'quote-washington', 'quoteb': 'quote-binladen',\
				  		   'd_art': 'prefer_art'})
	

	## -----
		
	for key in df.keys():
		df[key] = df[key].replace('', np.nan)
	
	missing_list = count_missing_per_sample(df)
	# create a histogram with missing values
	fig, ax = plt.subplots()
	plt.hist(missing_list)
	plt.xlabel("number of missing values")
	plt.savefig("missing_values.png")
	df = df.drop(df.index[np.where(np.array(missing_list)>30)[0]])	
	

	# list with binary featrures
	bin_list = ['session_date', 'runalone', 'flag-american', 'money-first', 'forbidden', 'allowed', 'disease-save-choseprob',
			'disease-kill-choseprob', 'omdimc3-pass', 'reciprocityothera', 'reciprocityotherb', 'reciprocityusa', 'reciprocityusb',
			'subject-international', 'exp-online', 'order', 'prefer_art']

	# add one-hot encoded features
	dummies_list = ["referrer", "expgender", "exprace", "compensation", "recruitment", "separatedornot", "age", 'ethnicity', 'flagsuppl-american', 'flagsuppl-republican', \
			    'flagsuppl-conservative', 'artwarm', 'art-good2bad', 'art-happy2sad', 'art-beautiful2ugly', 'art-delightful2disgusting', 'art-approach2avoid', \
			    'art-unafraid2afraid', 'math-good2bad', 'mathwarm', 'math-happy2sad', 'math-beautiful2ugly', 'math-delightful2disgusting', 'math-approach2avoid',\
			    'math-unafraid2afraid','imaginedexplicit1', 'imaginedexplicit2', 'imaginedexplicit3', 'imaginedexplicit4', 'major', 'politicalid', \
			    'quote-washington', 'quote-binladen', 'gamblerfallacya', 'gamblerfallacyb', 'sunkcosta','sunkcostb', 'sex', 'scalesorder', 'reciprocorder', 'diseaseforder',\
			    'quoteorder', 'flagprimorder', 'sunkcostorder', 'anchorinorder', 'allowedforder', 'gamblerforder', 'moneypriorder', 'imaginedorder',\
			    'anchoring1a', 'anchoring1b','anchoring2a', 'anchoring2b', 'anchoring3a', 'anchoring3b', 'anchoring4a', 'anchoring4b',
			    'scalesa', 'scalesb']\
			     + ['flagdv'+str(i) for i in range(1,9)] + ['sysjust'+str(i) for i in range(1,9)]

	for dum in dummies_list:
		df = pd.concat([df, pd.get_dummies(df[dum], prefix=dum+"_")], axis=1)
		
	df = df.drop(columns=dummies_list+['age__','artwarm__', 'mathwarm__', 'gamblerfallacya__', 'gamblerfallacyb__'])
	
	# substitute with NaN, the zero slices that may result from applying get_dummies() 
	# this step might take some time to compute
	feat_dict = create_dict(df)
	for feat in feat_dict:
		if len(feat_dict[feat]) > 1:
			for i in range(len(df)):
				if not any(df.iloc[i, feat_dict[feat]]):
					df.iloc[i, feat_dict[feat]] = [np.nan]*len(feat_dict[feat])
	

	# create tran/val/test split (80%/10%/10%)
	random_indexes = np.random.permutation(df.shape[0])
	train_indices  = random_indexes[:4437]
	val_indices    = random_indexes[4437:4437+951]
	test_indices   = random_indexes[4437+951:]

	df_train = df.iloc[train_indices]
	df_val   = df.iloc[val_indices]
	df_test  = df.iloc[test_indices]

	print("Final shape for Train/Val/Test datasets: {} -- {} -- {}".format(df_train.shape, df_val.shape, df_test.shape))
	
	with open("train_set.csv", 'w') as ftrain, open("val_set.csv", 'w') as fval, open("test_set.csv", 'w') as ftest:
		ftrain.write(df_train.to_csv(index=False))
		fval.write(df_val.to_csv(index=False))
		ftest.write(df_test.to_csv(index=False))
	
