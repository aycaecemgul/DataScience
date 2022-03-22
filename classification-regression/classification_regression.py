import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

project= int(input("Which project you would like to see? (1/2)"))


if(project==1):
#                                            PART 1 - REGRESSION

    #function that gets year as a parameter and returns predicted salary
    def predictSalary(predInput):
        test = np.array([predInput]).reshape(1, -1)
        predict = lr.predict(test)
        return predict

    dataset = pd.read_csv("salary.csv") #getting the data from csv file.

    #splitting the data.
    X=dataset[["experience"]]
    y=dataset[["salary"]]
    lr= linear_model.LinearRegression()
    lr.fit(X, y)

    #simulating the linear regression
    plt.scatter(X, y, color = "red")
    plt.plot(X, lr.predict (X), color ="blue")
    plt.xlabel("X : experience")
    plt.ylabel("Y : salary")
    plt.title("Relationship between experience - salary")
    agreement= input("Welcome to the regression project. Do you want to see the simulation? (Y/N): ")
    if(agreement=="Y"): plt.show()

    print("coefficients:",lr.coef_)
    print("Mean squared error : %.2f" % np.mean((lr.predict(X) - y) ** 2))
    value="Y"

    while(value!="N" or value!="n"):

        predInput = int(input("Enter the year(s) of experience:\n")) #getting the input
        print(int(predictSalary(predInput)))
        value=str(input("Do you want to calculate another one?(Y/N)"))
        if(value=="N" and value!="Y"):
            print("Goodbye.")
            exit()



elif(project==2):

#                                           PART 2 - CLASSIFICATION

    # This classification is about a face cream formula. The ingredients' percentage has an important role
    # in the formula for active ingredients to work. Formula must satisfy the customers' expectations
    # and show its benefits from inside out. Percentages and customers' feedback is given to the algorithm.
    # You can compare your future formulas to our statics provided by my algorithm.
    # All of this (problem&data) are made up by me!

    # input: ingredient percentages.
    # output: predict whether a formula is good or not.

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    # getting the dataset from the file
    dataset = pd.read_csv("data.csv")
    print("Welcome to the classification project.")
    print("Gathering the data..")
    # splitting the data for testing/teaching
    X = dataset.iloc[:, 0:8]
    y = dataset.iloc[:, 8]
    print("Splitting dataset..")

    # #creating the decision tree classifier
    clf= DecisionTreeClassifier(random_state=0)
    X_train, X_test , y_train , y_test = train_test_split(X,y, train_size = 0.6, test_size =0.4, random_state = 0, stratify =y)

    # #teaching
    clf.fit(X_train,y_train)
    print("Training the algorithm..")

    # getting the predictions of algorithm
    result= clf.predict(X_test)

    print("Algorithm has made its predictions.\n")

    print("Accuracy score:")
    print(accuracy_score(result, y_test))


    # getting the confusion matrix
    cm =confusion_matrix(y_test , result)
    print("Confusion Matrix :")
    print(cm)
    # cross validating, k=10
    cValScore = cross_val_score(clf , X , y, cv=10)
    print("10-Fold Cross Validation Score : ")
    print(cValScore)
    print("Goodbye.")
else:
    print("Incorrent value. Bye.")