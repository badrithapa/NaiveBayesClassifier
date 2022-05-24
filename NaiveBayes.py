#implement Gaussian Naive Bayes to make predictions
from csv import reader
from math import sqrt
from math import exp
from math import pi
from random import seed
from random import randrange

class GaussianNB:
    """
    Gaussian Naive Bayes supports continuous valued features and models each as conforming to a Gaussian (normal) distribution. This model can be fit by simply finding the mean and standard deviation of the points within each label, which is all what is needed to define such a distribution.
    """
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
        '''
        Load a csv file and turn it in t a list of lists (each list is a row of the csv file)
        ---------------
        Parameters
        ----------
        filename: string - the path to the csv file
        ------
        Returns
        --------
        dataset: list - each element is a list of lists (each list is a row of the csv file)
        '''
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
        '''
        Converts all the values in a column to floats.
        ---------------
        Parameters
        ----------
        dataset: list - each element is a list of lists (each list is a row of the csv file)
        column: int - the column to convert
        ------
        Returns
        --------
        None - the dataset is modified in place
        '''
        for row in dataset:
            row[column] = float(row[column].strip())

    #Split the dataset by cass vlaues
    def str_column_to_int(self, dataset, column):
        '''
        Converts all the values in a column, specifically the target column which contains categorical data, to integers.
        ----------------
        Parameters
        ----------
        dataset: list - each element is a list of lists (each list is a row of the csv file)
        column: int - the column to convert
        ------
        Returns
        --------
        None - the dataset is modified in place

        '''
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
        '''
        Separates the dataset by class values.
        ---------------
        Parameters
        ----------
        dataset: list - each element is a list of lists (each list is a row of the csv file)
        ------
        Returns
        --------
        separated: dict - class_value as keys and list of rows as values
        '''
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
        '''
        Calculates the mean of a list of numbers i.e. column of dataset
        ---------------
        Parameters
        ----------
        numbers: list - each element is a number i.e. column of dataset
        ------
        Returns
        --------
        mean: float - the mean of the list of numbers i.e. column of dataset
        '''
        return sum(numbers)/float(len(numbers))

    def stdev(self, numbers):
        '''
        Calculates the standard deviation of a list of numbers i.e. column of dataset
        ---------------
        Parameters
        ----------
        numbers: list - each element is a number i.e. column of dataset
        ------
        Returns
        --------
        stdev: float - the standard deviation of the list of numbers i.e. column of dataset
        '''
        # print(numbers)
        avg = self.mean(numbers)
        variance = sum([(x - avg)**2 for x in numbers])/float(len(numbers)-1)
        return sqrt(variance)

    #calculate mean and standard deviation and count for each column
    def summarize_dataset(self, dataset):
        '''
        Calculates the mean, standard deviation and count samples for each column of the dataset.
        ---------------
        Parameters
        ----------
        dataset: list - each element is a list of lists (each list is a row of the csv file)
        ------
        Returns
        --------
        summaries: list - list that contains tuples of (mean, standard deviation, count) samples for each column of the dataset
        '''
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*dataset)]
        del summaries[-1]
        # print(summaries)
        return summaries

    # summarize by class then calculate mean and standard deviation
    def summarize_by_class(self, dataset):
        '''
        Calculates the mean, standard deviation and count samples for each column of the dataset with respect to it's class.
        ---------------
        Parameters
        ----------
        dataset: list - each element is a list of lists (each list is a row of the csv file)
        ------
        Returns
        --------
        summaries: dict - class_value as keys and list of tuples of (mean, standard deviation, count) samples for each column of the dataset as values
        '''
        separated = self.split_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    # calculate the Gaussian probability distribution function
    def calculate_probability(self, x, mean, stdev):
        '''
        Calculates the probability of x for the Gaussian distribution with mean and standard deviation.
        ---------------
        Parameters
        ----------
        x: float - the value to calculate the probability for
        mean: float - the mean of the Gaussian distribution
        stdev: float - the standard deviation of the Gaussian distribution
        ------
        Returns
        --------
        probability: float - the probability of x for the Gaussian distribution with mean and standard deviation
        '''
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    # calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(self, summaries, row):
        '''
        Calculates the probabilities of predicting each class for a given row.
        ---------------
        Parameters
        ----------
        summaries: dict - class_value as keys and list of tuples of (mean, standard deviation, count) samples for each column of the dataset as values
        row: list - each element is a number a row is an instance of the data i.e. features of the data
        ------
        Returns
        --------
        probabilities: dict - class_value as keys and Gaussian probability as values
        '''
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
        '''
        Predicts the class for a given row.
        ---------------
        Parameters
        ----------
        summaries: dict - class_value as keys and list of tuples of (mean, standard deviation, count) samples for each column of the dataset as values
        row: list - each element is a number a row is an instance of the data i.e. features of the data
        ------
        Returns
        --------
        best_label: str - the predicted class for the given row(features)
        '''
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    #Naive Bayes algorithm
    def naive_bayes_classifier(self, train, test):
        '''
        The main classifier algorithm.
        ---------------
        Parameters
        ----------
        train: list - each element is a list of lists (each list is a row of the csv file)
        test: list - each element is a list of lists (each list is a row of the csv file)
        ------
        Returns
        --------
        predictions: list - each element is a predicted class for the given row(features)
        '''
        predictions = list()
        summaries = self.summarize_by_class(train)
        for row in test:
            label = self.predict(summaries, row)
            predictions.append(label)
        return predictions

    # k folds cross validation
    def cross_validation_split(self, dataset, n_folds):
        '''
        Splits the dataset into k folds for validation.
        ---------------
        Parameters
        ----------
        dataset: list - each element is a list of lists (each list is a row of the csv file)
        n_folds: int - number of folds
        ------
        Returns
        --------
        dataset_split: list - each element is a list of lists (each list is a row of the csv file)
        '''
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
        '''
        Calculates the accuracy percentage.
        ---------------
        Parameters
        ----------
        actual: list - each element is an actual class of the data
        predicted: list - each element is a predicted class for the data
        ------
        Returns
        --------
        accuracy: float - the accuracy percentage
        '''
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        '''
        Validates the algorithm using a cross validation split.
        ---------------
        Parameters
        ----------
        dataset: list - each element is a list of lists (each list is a row of the csv file)
        algorithm: function - the algorithm to evaluate
        n_folds: int - number of folds
        *args: list - additional arguments for the algorithm
        ------
        Returns
        --------
        scores: list - each element is the accuracy percentage for the algorithm
        '''
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
    summaries = gnb.summarize_by_class(dataset)
    # print(summaries)
    row = [5.7, 2.9,4.2,1.3]
    label = gnb.predict(summaries, row)
    print('Data=%s, Predicted: %s' % (row, label))
    # n_folds = 5
    # scores = gnb.evaluate_algorithm(dataset, gnb.naive_bayes_classifier, n_folds)
    # print('Scores: %s' % scores)
    # print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

