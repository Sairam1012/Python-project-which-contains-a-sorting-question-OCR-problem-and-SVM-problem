#!/usr/bin/env python
# coding: utf-8

# Question 01
# Sorting the files using filenames

# In[4]:


import re 

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


# In[5]:


s = set(['mallika_1.jpg', 'dog005.jpg', 'grandson_2018_01_01.png', 'dog008.jpg', 'mallika_6.jpg', 'grandson_2018_5_23.png', 'dog01.png', 'mallika_11.jpg', 'mallika2.jpg', 'grandson_2018_02_5.png', 'grandson_2019_08_23.jpg', 'dog9.jpg', 'mallika05.jpg'])
for x in sorted_nicely(s):
    print(x)


# Question 02

# Here I have used SVM technique to analyse the data.
# Importing the dataset and required pacakages.

# In[1]:


from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv("Q2_data_set.csv")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


# Splitting data into train and test data.

# In[2]:


X, y = make_blobs(n_samples=125, centers=2, cluster_std=0.60, random_state=0)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=20, random_state=0)


# In[3]:


plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='winter')


# 

# In[4]:


svc=SVC(kernel='linear')
svc.fit(train_X, train_y)


# In[5]:


plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='winter')
ax=plt.gca()
xlim=ax.get_xlim()
ax.scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap='winter', marker='s')

w = svc.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - (svc.intercept_[0] / w[1])
plt.plot(xx, yy)
plt.show()


# In[ ]:





# Question 3

# In[12]:


conda install -c conda-forge pypdf2


# In[13]:


conda install -c conda-forge poppler


# In[14]:


conda install -c conda-forge pytesseract


# In[2]:


import os
import tempfile
from pdf2image import convert_from_path


# In[3]:


filename = 'Q3_Image_File.pdf'

with tempfile.TemporaryDirectory() as path:
     images_from_path = convert_from_path(filename, output_folder=path, last_page=1, first_page =0)

base_filename  =  os.path.splitext(os.path.basename(filename))[0] + '.jpg'      

save_dir = 'Q3_Image_File.jpg'

for page in images_from_path:
    page.save(os.path.join(save_dir, base_filename), 'JPEG')


# In[8]:


from PIL import Image
import pytesseract, re
f = "Q3_Image_File.jpg"
t = pytesseract.image_to_string(Image.open(f))
m = re.findall(r"Tumkur Road, Availability, Industrial/warehouse, Nelamangla, Dobespet. 1 to 100 acres. 1000 to 100000, sqft. Naik. 9141326819+", t)
if m:
    print(m[0])


# In above statement , I have extracted text from image using tesseract and then i stores it into t.
# Now below statement,we are printing 't' which contains the text extracting.

# In[9]:


t


# In[10]:


search_string = 'Warehouse'
print(search_string in t)


# In[ ]:




