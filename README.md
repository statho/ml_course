# Machine Learning Project - Missing Data Imputation

This is the final project for the Machine Learning course, in the CS department of Rutgers University

The code for this project is written in python 3 (version 3.7.2).
All models were implemented from scratch -- the only libraries used were for data handling as encouraged by the instructor.

## Preprocessing step
Run
```shell
python preprocessing.py <name_of_csv_file>
```
 		
Preprocess and split dataset into training, validation and testing. Files produced train_set.csv, val_set.csv and test_set.csv


## Training the model
Run
```shell
python train_model.py
```

Train the model using a list of configurations. Produce necessary files that track model the model's performance and save the best model in a .pkl file


## Fill in missing values using the model
Run
```shell
python fill_values.py <name_of_model_to_use>
```


*** The implementation of the Neural Network is contained in model.py,
	the softmax, the loss function and its derivative are implemented in losses.py
	and the activation functions are implemented in activations.py
