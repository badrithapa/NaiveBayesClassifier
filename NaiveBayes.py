#implement Gaussian Naive Bayes to make predictions
from csv import reader
from math import sqrt
from math import exp
from math import pi
from random import seed
from random import randrange

class GaussianNB:
    def __init__(self):
        pass
        # self.filename = path_to_csv
        # self.dataset = self.load_csv(self.filename)
        # self.str_column_to_float(self.dataset, -1)
        # self.str_column_to_int(self.dataset, -1)
        # self.unique = self.str_column_to_int(self.dataset, -1)
        # self.class_values = list(self.unique)
        # self.class_counts = [len(list(filter(lambda x: x == value, self.dataset))) for value in self.unique]
        # self.mean = list()
        # self.stdev = list()
        # self.gaussian_naive_bayes_classifier()

    #load a csv file
    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    # Convert String class to float
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())

    #Split the dataset by cass vlaues
    def str_column_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
            print('[%s] => %d'  % (value, i))
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    #split the dataset b class values, return a dictionary
    def split_by_class(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
        return separated

    #calculate mean and standard deviation
    def mean(self, numbers):
        return sum(numbers)/float(len(numbers))

    def stdev(self, numbers):
        print(numbers)
        avg = self.mean(numbers)
        variance = sum([(x - avg)**2 for x in numbers])/float(len(numbers)-1)
        return sqrt(variance)

    #calculate mean and standard deviation and count for each column
    def summarize_dataset(self, dataset):
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*dataset)]
        del summaries[-1]
        # print(summaries)
        return summaries

    # summarize by class then calculate mean and standard deviation
    def summarize_by_class(self, dataset):
        separated = self.split_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    # calculate the Gaussian probability distribution function
    def calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    # calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(self, summaries, row):
        probabilities = dict()
        total_rows = sum([summaries[label][0][2] for label in summaries])
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities

    # predict the class for a given row
    def predict(self, summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    #Naive Bayes algorithm
    def naive_bayes_classifier(self, train, test):
        predictions = list()
        summaries = self.summarize_by_class(train)
        for row in test:
            label = self.predict(summaries, row)
            predictions.append(label)
        return predictions

    # k folds cross validation
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores



if __name__ == "__main__":
    # Test calculating class probabilities
    gnb = GaussianNB()
    dataset = gnb.load_csv('iris.csv')
    for i in range(len(dataset[0])-1):
        gnb.str_column_to_float(dataset, i)
    gnb.str_column_to_int(dataset, len(dataset[0])-1)
    n_folds = 5
    scores = gnb.evaluate_algorithm(dataset, gnb.naive_bayes_classifier, n_folds)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

