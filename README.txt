# TFIDF Model 

File #1 : NLU-TFIDF-demo.ipynb (inferencing test data)
File #2 : NLU-TFIDF.ipynb (training and evaluation)

## Code Structure for #1
- Data Preprocessing: reads CSV files containing text data, preprocesses it by removing non-alphabetical characters, converting to lowercase, tokenizing, removing stopwords, and lemmatizing, then converts labels to categorical format.
- Model Definition: Implementation of the TFIDF model. It uses the preprocessed data, generates TF-IDF features, and saves/loads the TF-IDF vectorizer based on mode ('train' or 'test').
- Evaluation: Functions to dump the predictions file.

## Code Structure for #2 
The codebase is divided into several sections:
- Data Preprocessing: reads CSV files containing text data, preprocesses it by removing non-alphabetical characters, converting to lowercase, tokenizing, removing stopwords, and lemmatizing, then converts labels to categorical format.
- Model Definition: Implementation of the TFIDF model. It uses the preprocessed data, generates TF-IDF features, and saves/loads the TF-IDF vectorizer based on mode ('train' or 'test').
- Training: logistic_regression_train trains a logistic regression model using TF-IDF features extracted from the training data, saves the trained model to a file, and prints a completion message.
- Evaluation: Functions to test the model and compute accuracy, precision, recall, F1-score, confusion matrices, processing times, macro and weighted avg. This also dumps the predictions file.


## Attributions, Data Sources and Pre-trained Embeddings
- The TFIDF approach is based on the research paper "A Comparative Study on TF-IDF feature Weighting Method and its Analysis using Unstructured Dataset" : https://arxiv.org/abs/2308.04037 and COMP34711: Natural Language Processing module in the Final Year of the Computer Science course at the University of Manchester.
- The code is adapted from this GitHub repository : https://github.com/vineeths96/Natural-Language-Inference/tree/master

## Pre-trained Model and files (https://livemanchesterac-my.sharepoint.com/:f:/g/personal/aditya_student_manchester_ac_uk/EqhzWT1sRg1EpABWsoEJRqUBdIuICWoRqjE0f1ZGcf_QzQ?e=7b8KGe) 
Download the following files from the link:
- TFIDF VECTORIZER : OneDriveDirectory/TFIDF/TFIDF.pickle
  This is the serialized TF-IDF vectorizer for transforming text data into numerical features
- LF Model : OneDriveDirectory/TFIDF/LF.pickle
  This is the trained logistic regression model


## Usage of the demo file:
To use the code, do the following, modify the following 3 things in the Config block [MANDATORY]:
       a) config_testing.testdata : Data under inference
       b) config_testing.vectorizer: Vectorizer file downloaded from OneDrive
       c) config_testing.model: Model file downloaded from OneDrive 


## Important Notes
1. Please ensure that if you're utilizing sub-directories for the files, you create them manually beforehand, as the code doesn't check if the directories exist.
2. It is mandatory to supply all three files mentioned in the above section for the code to work. 
3. Ensure that all filepaths contain complete filenames with their respective extensions.
4s. The output file is generated in the working directory with the name predictions.csv


## Adaptation of Publicly Available Code
Below, we outline the modifications made to the publicly available code suit the specific requirements of our task:

- Feature Generation: The original code provided multiple options for generating features from TFIDF representations. After thorough experimentation, we opted for option 5, which involved concatenating TFIDF vectors for both sentence1 and sentence2. This choice was made based on experiments to ensure good performance for our task.
- Class Modification: The original code was designed to handle three classes: "entailment," "neutral," and "contradiction." We adapted it to focus solely on "entailment" and "contradiction" for our specific task.
- Additional Metrics: We extended the functionality of the code by incorporating new evaluation metrics beyond accuracy. These included confusion matrix, classification report, F1 score, precision, and recall.
- Code Refinement: We conducted general code cleanup to ensure compatibility with our coursework requirements. This involved adding comments, removing unnecessary code, and adjusting input/output formats/files as needed.
- Demo Code Modification: We modified the demo code to support inference on unseen data without relying on labels


###############################################################################################
###############################################################################################
###############################################################################################


# ESIM Model for Natural Language Inference (Enhanced LSTM)

File #1 : NLU-ESIM-demo.ipynb (inferencing test data)
File #2 : NLU-ESIM.ipynb (training and evaluation)


## Code Structure for #1
The codebase is divided into several sections:
- Model Definition: Implementation of the ESIM model with its various layers and attention mechanisms.
- Data Preprocessing: Code for reading raw data, building word dictionary, and generating preprocessed pickle files for the data and embedding matrix. 
- Testing: Loading the saved model and dumping the predictions file with the results.

## Code Structure for #2 
The codebase is divided into several sections:
- Model Definition: Implementation of the ESIM model with its various layers and attention mechanisms.
- Data Preprocessing: Code for reading raw data, building word dictionary, and generating preprocessed pickle files for the data and embedding matrix. 
- Training: Functions to train the ESIM model. 
- Evaluation: Functions to test the model and compute accuracy, precision, recall, F1-score, confusion matrices, processing times, macro and weighted avg. This also dumps the predictions file.

## Attributions, Data Sources and Pre-trained Embeddings
- The model uses pre-trained GloVe (Global Vectors for Word Representation) embeddings from Stanford NLP projects: (https://nlp.stanford.edu/projects/glove : Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): `glove.840B.300d.zip`)
- The ESIM model is based on the research paper "Enhanced LSTM for Natural Language Inference" : https://arxiv.org/abs/1609.06038 and https://paperswithcode.com/method/esim).
- The code is adapted from this GitHub repository : https://github.com/coetaur0/ESIM

## Pre-trained Model and files (https://livemanchesterac-my.sharepoint.com/:f:/g/personal/aditya_student_manchester_ac_uk/EqhzWT1sRg1EpABWsoEJRqUBdIuICWoRqjE0f1ZGcf_QzQ?e=7b8KGe) 
To use pre-trained models, download the following checkpoint files from the link:
- MODEL : OneDriveDirectory/Enhanced LSTM/esim_model.tar  
  This contains the best-trained model with pre-trained embeddings and weights.
- TRAINING FILES [OPTIONAL]: OneDriveDirectory/Enhanced LSTM/Files dumped while training/*
  Contains weights for all 10 epochs, preprocessed embedding matrix, preprocessed input files, preprocessed word dictionaries. These can be used these to test different stages of the model's development.

## Usage of the demo file:
To use the code, do the following, modify the following 3 things in the Config block [MANDATORY]:
       a) config_preprocessing.training_input_data_file : Original Training File (required for generating word dictionary during runtime) OneDriveDirectory/train.csv
       b) config_preprocessing.test_input_data_file: Data under inference
       c) config_testing.model: Model Weight downloaded from OneDrive (esim_model.tar )


## Important Notes
1. If you're not storing the files on Google Drive, simply comment out the code responsible for mounting it.
2. Please ensure that if you're utilizing sub-directories for the files, you create them manually beforehand, as the code doesn't check if the directories exist.
3. It is mandatory to supply all three files mentioned in the above section for the code to work. 
4. Please do not extract the model file, leave it in its original .tar format when loading into the environment.
5. Ensure that all filepaths contain complete filenames with their respective extensions.
6. The output file is generated in the working directory with the name predictions.csv


## Adaptation of Publicly Available Code
Below, we outline the modifications made to the publicly available code suit the specific requirements of our task:

- GloVe Embeddings: Initially, we attempted to create our own GloVe embeddings based on our training dataset. However, after testing, we found that the preprovided embeddings from Stanford yielded better results. We used the GloVe implementation from Stanford's GitHub repository for this purpose.
- Class Modification: The original code was designed to handle three classes: "entailment," "neutral," and "contradiction." We adapted it to focus solely on "entailment" and "contradiction" for our specific task.
- Additional Metrics: We extended the functionality of the code by incorporating new evaluation metrics beyond accuracy. These included confusion matrix, classification report, F1 score, precision, and recall.
- Code Refinement: We conducted general code cleanup to ensure compatibility with our coursework requirements. This involved adding comments, removing unnecessary code, and adjusting input/output formats/files as needed.
- Demo Code Modification: We modified the demo code to support inference on unseen data without relying on labels


###############################################################################################


Contributors
* Aditya Agarwal aditya@student.manchester.ac.uk (The University of Manchester, Manchester, UK)
* Varun Shankar varun.shankar@student.manchester.ac.uk (The University of Manchester, Manchester, UK)
