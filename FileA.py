from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


data = load_iris()


x= data['data']
y= data['target']


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0,random_state=0)

clf = tree.DecisionTreeClassifier()
clf.fit(train_x,train_y)

pred_y = clf.predict([test_x])



accuracy = accuracy_score(test_y, pred_y)
print(accuracy)
pred_y_train = clf.predict(train_x)
train_accuracy =  accuracy_score(train_y, pred_y_train)

print(data['target_names'][train_y])