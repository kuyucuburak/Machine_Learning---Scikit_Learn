from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy


def main():
    numpy.set_printoptions(suppress=True, precision=False, threshold=numpy.inf)
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=DeprecationWarning)


    fileName = "dataset.txt"
    file = open(fileName, "r")
    fileContent = file.read()

    """ I changed letters to 0 and 1 to load text file as a numpy array."""
    fileContent = fileContent.replace("M", "0")
    fileContent = fileContent.replace("B", "1")

    file = open("temp.txt", "w")
    file.write(fileContent)

    applyModel(0)
    applyModel(1)
    applyModel(2)


def applyModel(model):
    data = numpy.loadtxt("temp.txt", delimiter=",")
    if model == 0:
        numpy.random.seed(90)
    elif model == 1:
        numpy.random.seed(30)
    else:
        numpy.random.seed(50)
    numpy.random.shuffle(data)

    trainError = []
    testError = []
    N = 10
    xTicks = []
    for loop in range(1, N+2):
        subN = int(len(data) / N) * loop
        if subN > len(data):
            subN = len(data)
        elif subN == len(data):
            break

        subTrainError = []
        subTestError = []
        results = data[0:subN, 1:2]
        nData = data[0:subN, 2:]
        kFold = KFold(n_splits=10)
        i = 0
        for trainIndex, testIndex in kFold.split(nData):
            trainData, testData = nData[trainIndex], nData[testIndex]
            trainResult, testResult = results[trainIndex], results[testIndex]
            trainResult, testResult = numpy.array(trainResult).ravel(), numpy.array(testResult).ravel()

            if model == 0:
                parameters = {'max_depth': (10, 100)}
                svc = RandomForestClassifier()
            elif model == 1:
                parameters = {'max_iter': (10, 100)}
                svc = MLPClassifier(solver='lbfgs', shuffle=False)
            else:
                parameters = {'gamma': (0.01, 0.001), 'C': [1, 10]}
                svc = SVC()


            svc = GridSearchCV(svc, parameters, cv=2)

            subTrainError.append(100 - (svc.fit(trainData, trainResult).score(trainData, trainResult)) * 100)
            prediction = svc.predict(testData)

            subTestError.append(100 - (numpy.sum(testResult == prediction) / len(testResult) * 100))
            i = i + 1

        trainError.append(numpy.mean(subTrainError))
        testError.append(numpy.mean(subTestError))
        xTicks.append(subN)

    plt.xticks(numpy.linspace(0, 10, N), xTicks)
    plt.plot(trainError, color='red')
    plt.plot(testError, color='blue')
    plt.show()


if __name__ == "__main__":
    main()
