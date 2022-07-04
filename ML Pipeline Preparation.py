#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[153]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


# In[144]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName',engine)
X = df.message
Y = df.iloc[:,4:]


# In[212]:


#https://knowledge.udacity.com/questions/726905

df.related.unique()


# In[230]:


for column in Y.columns:
    print(column,df[column].unique())


# In[215]:


df=df[df.related!=2]


# In[216]:


df.shape


# ### 2. Write a tokenization function to process your text data

# In[145]:


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    return tokens


# In[146]:


import nltk
nltk.download(["punkt","stopwords","wordnet"])


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[154]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[217]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
pipeline.fit(X_train,y_train)


# In[238]:


y_pred2=pd.DataFrame(y_pred,columns=y_test.columns)
y_pred2


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[263]:


y_pred


# In[254]:


y_pred = pipeline.predict(X_test)
target_names = list(Y.columns)
print(classification_report(y_test,y_pred,target_names =target_names))


# In[246]:


y_test.shape


# In[259]:


y_pred2=pd.DataFrame(y_pred,columns= y_test.columns)


# In[261]:


y_pred2.shape


# In[262]:


target_names = list(Y.columns)
print(classification_report(y_test,y_pred,target_names =target_names))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[253]:


parameters = {
    'clf__estimator__n_estimators': [5],
    'clf__estimator__min_samples_split': [2]
}

cv = GridSearchCV(pipeline, param_grid=parameters)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:





# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# ### 9. Export your model as a pickle file

# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




