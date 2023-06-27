import numpy
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
data = pandas.read_csv('Breast-cancer.csv')
X = numpy.array(data.iloc[2:,:-1])
y = numpy.array(data.iloc[2:,-1:].values.ravel())
X = StandardScaler().fit(X).transform(X)
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.21, random_state = 18)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(XTrain, yTrain)
predict = knn.predict(XTest)
print('Accuracy :', accuracy_score(predict, yTest))
print('f1Score :' ,f1_score(yTest, predict))
print('hi')
