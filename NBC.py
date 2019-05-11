import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt


class NaiveBayes:
    def __init__(self, laplace=0):
        self.PY = None
        self.PXY = None
        self.YKeys = None
        self.XKeys = None
        self.laplace = laplace

    def populate_py(self, y):
        self.YKeys = np.unique(y)
        PY = dict.fromkeys(self.YKeys, 0)
        for i in y:
            PY[i] += 1
        self.PY = PY

    def populate_pxy(self, x, y):
        self.XKeys = np.unique(x)
        samplePXY = dict.fromkeys(self.YKeys)
        PXY = []
        for sample in x:
            for i in self.YKeys:
                samplePXY[i] = dict.fromkeys(self.XKeys, 0)
            for i in range(len(y)):
                samplePXY[y[i]][sample[i]] += 1

            PXY.append(samplePXY.copy())
        self.PXY = PXY

    def fit(self, xu, yu):
        NaiveBayes.populate_py(self, yu)
        NaiveBayes.populate_pxy(self, xu, yu)

    def calculate_py(self):
        Yprobability = dict.fromkeys(np.unique(self.YKeys))
        for j, i in self.PY.items():
            Yprobability[j] = i / sum(self.PY.values())
        return Yprobability

    def calculate_px(self, x):
        Xprobability = dict.fromkeys(self.YKeys)
        for i in self.YKeys:
            Xprobability[i] = dict.fromkeys(self.XKeys, 0)
        for i in self.YKeys:
            for j in self.XKeys:
                numerator = self.PXY[x][i][j] + 1 * self.laplace
                denominator = self.PY[i] + 1 * self.laplace
                Xprobability[i][j] = numerator / denominator
        return Xprobability

    def calculate_pxy(self, x):
        Yprobability = NaiveBayes.calculate_py(self)
        Xprobability = NaiveBayes.calculate_px(self, x).copy()
        for i in self.YKeys:
            for j in self.XKeys:
                Xprobability[i][j] = Xprobability[i][j] * Yprobability[i]
        return Xprobability

    def predict_proba(self, sample):
        Yprobability = NaiveBayes.calculate_py(self)
        Xprobability = []
        for i in range(len(sample)):
            Xprobability.append(NaiveBayes.calculate_px(self, i).copy())
        prob_list = dict.fromkeys(np.unique(self.YKeys), 0)
        for i in self.YKeys:
            prob = 0
            for j in range(len(self.PXY)):
                if j == 0:
                    prob = Xprobability[j][i][sample[j]]
                else:
                    prob = prob * Xprobability[j][i][sample[j]]
            prob = prob * Yprobability[i]
            prob_list[i] = prob
        return max(prob_list, key=prob_list.get), prob_list


def bayesError(y, pred_y):
    match_count = 0
    for i in range(len(pred_y)):
        if pred_y[i] == y[i]:
            match_count = match_count + 1
    match_percentage = match_count / len(pred_y) * 100
    return match_percentage


def divide_into_learn_and_test_sets(y, x, proportion):
    data = np.vstack((x, y))
    data = np.rot90(data)
    data = np.random.permutation(data)
    data = np.rot90(data, k=-1)
    z = int(round(np.size(data, 1) * proportion))
    return data[-1, :z], data[:-1, :z], data[-1, z:], data[:-1, z:]


def main():
    data = arff.loadarff("zoo.arff")
    df = pd.DataFrame(data[0])
    X = np.array(df)
    X = X.astype('U13')

    Y = X[:, -1]
    X = np.hstack((X[:, 1:13], X[:, 14:-1]))
    X = np.rot90(X)

    alfa = (np.linspace(0.05, 0.95, 100))
    repeat_number = 50
    NB = NaiveBayes()
    NBwL = NaiveBayes(laplace=1)

    accuracy_list = []
    accuracy_listwL = []

    plt.figure()
    for k in range(0, np.size(alfa)):
        print("************************")
        print(k)
        guess_list = []
        accuracy = []
        guess_listwL = []
        accuracywL = []

        ylearn, xlearn, ytest, xtest = divide_into_learn_and_test_sets(Y, X, alfa[k])
        for j in range(0, repeat_number):
            NB.fit(xlearn, ylearn)
            NBwL.fit(xlearn, ylearn)
            for z in range(len(ytest)):
                guess, prob_values = NB.predict_proba(xtest[:, z])
                guess_list.append(guess)
                guess, prob_values = NBwL.predict_proba(xtest[:, z])
                guess_listwL.append(guess)
            #final_guess.append(max(guess_list, key=guess_list.count))
            accuracy.append(bayesError(guess_list, ytest))
            accuracywL.append(bayesError(guess_listwL, ytest))
            #print(guess_list)
        print(sum(accuracy) / len(accuracy))
        accuracy_list.append(sum(accuracy) / len(accuracy))
        print(sum(accuracywL) / len(accuracywL))
        accuracy_listwL.append(sum(accuracywL) / len(accuracywL))

    plt.plot(alfa[:], accuracy_list, 'r-', label=' without laplacian smoothing')
    plt.plot(alfa[:], accuracy_listwL, 'b-', label=' with laplacian smoothing')
    plt.xlabel('Alfa')
    plt.ylabel('Accuracy[%]')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
