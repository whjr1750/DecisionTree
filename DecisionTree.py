import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics 

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
# from sklearn import tree


column_names = ['Pregnant' , 'Glucose' , 'BP' , 'Skin', 'Insulin' , 'BMI' , 'Pedigree' , 'Age' , 'Label']
df = pd.read_csv('csv/diabetes.csv' , names = column_names).iloc[1:]

print(df)

features = ['Pregnant' , 'Insulin' , 'BMI' , 'Age' , 'Glucose' , 'BP' , 'Pedigree']
X = df[features]

y = df['Label']

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.25 , random_state = 0)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train , y_train)

y_prediction = clf.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_prediction)
print(accuracy)

#Storing Data from the decision tree classifier as a text
dot_data = StringIO()
export_graphviz(clf , out_file = dot_data , filled = True , rounded = True , special_characters = True , feature_names =  features , class_names = ['0','1'])
print(dot_data.getvalue())

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Diabetes2.png')

Image(graph.create_png())