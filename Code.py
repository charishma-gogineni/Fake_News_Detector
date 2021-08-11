# Importing required libraries
import pandas as pd
import numpy as np

# Importing labelled dataset
data = pd.read_csv('data.csv')

# Removing the columns which are not required
data = data.drop(['URLs'],axis=1)
data = data.dropna()

data.head() #Viewing the data

# Defining dependent(y) and independent(x) variables
x= data.iloc[:,:-1].values
y= data.iloc[:,-1].values

# Using Bag of Words method to vectorize dataset
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
matrix_body = cv.fit_transform(x[:,1]).todense()
matrix_head = cv.fit_transform(x[:,0]).todense()

#Viewing matrices for headline and body of the news
matrix_head
matrix_body

x_mat = np.hstack((matrix_head, matrix_body)) #combining the head and body matrix into one matrix of independent variable

# Splitting training and testing dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_mat, y, test_size=0.2, random_state=0)


# Decision Tree Classification Model

from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train)
dtc.score(x_test, y_test) #viewing model score
y_pred= dtc.predict(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


#Logistic Regression Model

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(max_iter=5000)
lr.fit(x_train, y_train)
lr.score(x_test, y_test) #viewing model score
y_pred= lr.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
