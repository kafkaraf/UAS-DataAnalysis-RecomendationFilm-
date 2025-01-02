#!/usr/bin/env python
# coding: utf-8

# In[2]:


import joblib

def predict(data):
    clf = joblib.load('knn_model.sav')
    return clf.predict(data)


# In[ ]: