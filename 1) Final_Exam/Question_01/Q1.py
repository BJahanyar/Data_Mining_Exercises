import pydotplus
import pandas as pd
from io import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cols = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','age','skin','diab_pred']
mydata =  pd.read_csv("Files/Diabetes_Diagnosis.csv")

def Change(diabetes):
  if diabetes == True:
    diabetes='1'
  else:
    diabetes='0'
  return diabetes
mydata.diabetes = mydata.diabetes.apply(Change)
y = mydata.diabetes
x = mydata[cols]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Files/Q1_DecisionTree.png')
Image(graph.create_png())

