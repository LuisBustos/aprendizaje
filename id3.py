#! /usr/bin/python
# -*- coding: utf-8 -*-

# Universidad Nacional Autónoma de México
# Asignatura: Aprendizaje
# Programa: Algoritmo ID3
# Descripción: implementación básica del algoritmo ID3 para crear
# árboles de decisión
#
# Autores:
#       Luis Alberto Oropeza Vilchis
#		Luis Enrique Bustos Ramírez

from __future__ import division  # float division
from math import log
from random import randint
import csv  # To read data
import argparse
import re


# To implement tree as data structure
class Node():

    def __init__(self, label, parent=None, depth=0):
        self.label = label
        self.depth = depth
        self.childs = list()

    def add_child(self, condition, node):
        self.childs.append((condition, node))

    def show(self, condition=''):
        if self.depth == 0:
            print('\t==== Generated Tree ====')
            print('({0})'.format(self))
        else:
            print('|{0}{1}{2}]--({3})'.format('\t\t'*(self.depth-1),
                                                '|--[' if self.depth != 1 else '--[',
                                                condition, self))
        for n in self.childs:
            n[1].show(n[0])
            print('|')

    def __str__(self):
        return self.label

    def evaluate(self, data):
        if len(self.childs) == 0:  # Leaf node
            return self.label
        for child in self.childs:
            if re.match(r'[<|>]=?[0-9]+$', child[0]):
                if re.match(r'[<|>]=?[0-9]+$', data[self.label]):
                    if data[self.label] == child[0]:
                        return child[1].evaluate(data)
                else:
                    if eval('{0}{1}'.format(data[self.label], child[0])):
                        return child[1].evaluate(data)
            elif re.match(r'^[a-zA-Z]+$', child[0]):
                if data[self.label] == child[0]:
                    return child[1].evaluate(data)


# Class to encapsulate problem data
# filename: file to obtain the data
# target: index of column to be classfied, default = lastest column
class CSVData():

    def __init__(self, filename=None, target=-1):
        self.target = target
        self.attributes = list()
        self.data = list()
        self.test = list()
        self.values = dict()
        if filename != None:
            self.filename = filename
            self.load_data()
            self.obtain_all_values()

    # Loads attributes and rows from csv file
    def load_data(self):
        with open(self.filename) as csvfile:
            data = csv.DictReader(csvfile)
            self.attributes = data.fieldnames
            self.data = [x for x in data]
            self.test.append(self.data.pop(randint(0, self.size()-1)))

    # Gets the values from the attributes
    def obtain_all_values(self):
        for attr in self.attributes:
            self.values[attr] = list(set(x[attr] for x in self.data))

    # Returns values from an attribute
    def get_attribute_values(self, attr):
        return self.values[attr]

    # Creates a subset data from a particular value of an attribute
    def filter(self, attribute, value):
        new_attrs = filter(lambda x: x != attribute, self.attributes)
        temp = CSVData()
        temp.set_target(self.target)
        temp.set_data(list(filter(lambda x: x[attribute] == value, self.data)))
        temp.set_attributes(new_attrs)
        return temp

    def size(self):
        return len(self.data)

    def get_attributes(self):
        return self.attributes

    def get_rows(self):
        return self.data

    def get_target(self):
        return self.attributes[self.target]

    def get_target_values(self):
        return self.values[self.get_target()]

    def get_row(self, index):
        return self.data[index]

    def get_num_attributes(self):
        return len(self.attributes)

    def set_data(self, data):
        self.data = data

    def set_attributes(self, attrs):
        self.attributes = attrs
        self.obtain_all_values()

    def set_target(self, t):
        self.target = t

    def get_test_data(self):
        return self.test


# Class to encapsulate id3 algorithm operations
# data: instance of CSVData
class ID3():

    def __init__(self, data):
        self.data = data

    def run(self):
        return self.id3(self.data)

    def entropy(self, *args):
        args = filter(lambda x: x > 0, args) # to avoid log(0) error
        total = float(sum(args))
        return sum(map(lambda x: -x/total*log(x/total, 2), args))

    def calc_target_entropy(self, data):
        target = data.get_target()
        results = [data.filter(target, x).size() for x in data.get_target_values()]
        return self.entropy(*results)

    def calc_attr_entropy(self, data, attr):
        result = 0.0
        total = data.size()
        for value in data.get_attribute_values(attr):
            subset_data = data.filter(attr, value)
            len_data_value = subset_data.size()
            result += (len_data_value/total)*self.calc_target_entropy(subset_data)
        return result

    def calc_attr_gain(self, data, attr, entropy_set):
        return entropy_set - self.calc_attr_entropy(data, attr)

    def id3(self, data, depth=0):
        entropy_set = self.calc_target_entropy(data)
        if entropy_set == 0:
            return Node(data.get_target_values()[0], depth=depth)
        if data.get_num_attributes() == 1:
            return Node('Not decided', depth=depth)
        gains = [(x, self.calc_attr_gain(data, x, entropy_set)) for x in data.get_attributes()[:-1]]
        winner = max(gains, key=lambda x: x[1])
        tree = Node(winner[0], depth=depth)
        for value in data.get_attribute_values(winner[0]):
            tree.add_child(value, self.id3(data.filter(winner[0], value), depth+1))
        return tree


# Class to measure the performance of the algorithm.
# tree: tree created by ID3 class
# data: instance of CSVData
# verbose: to print all information of evaluation
class PerformanceTest():

    def __init__(self, tree, data, verbose=False):
        self.tree = tree
        self.data = data
        self.verbose = verbose
        self.tp = 0  # True Positive
        self.tn = 0  # True Negative
        self.fp = 0  # False Positive
        self.fn = 0  # False Negative
        self.calculate_measures()

    def calculate_measures(self):
        counter = 0
        if self.verbose:
            print('\n\t==== Analyzing the data ====')
        for r in self.data.get_rows():
            value = r[self.data.get_target()]
            prediction = self.tree.evaluate(r)
            if re.match(r'[sS][ií]|\+', value):
                if prediction == value:
                    self.tp += 1
                else:
                    self.fn += 1
            elif re.match(r'[nN]o|-', value):
                if prediction == value:
                    self.tn += 1
                else:
                    self.fp += 1
            if self.verbose:
                print('Value : {0} ---- Prediction : {1}'.format(value, prediction))

    def true_positive_rate(self):
        return self.tp / (self.tp + self.fn)

    def false_positive_rate(self):
        return self.fp / (self.fp + self.tn)

    def precision(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    def get_size_samples(self):
        return self.tp + self.fp + self.tn + self.fn

    def print_statistics(self):
        print('\n\t==== Statistics ====')
        print('True Positive: {0}'.format(self.tp))
        print('True Negative: {0}'.format(self.tn))
        print('False Positive: {0}'.format(self.fp))
        print('False Negative: {0}'.format(self.fn))
        print('Precision: {0:0.2f}%'.format(self.precision()*100))
        print('True Positive Rate: {0}'.format(self.true_positive_rate()))
        print('False Positive Rate: {0}'.format(self.false_positive_rate()))

    def print_test(self):
        print('\n\t==== Evaluating test data ====')
        test = self.data.get_test_data()[0]
        prediction = self.tree.evaluate(test)
        print('Data: {0}'.format(test))
        print('Prediction: {0}'.format(prediction))



def main():
    parser = argparse.ArgumentParser(description='ID3 algorithm')
    parser.add_argument('file', metavar='file', type=str, help='File path to retrieve the data')
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    args = parser.parse_args()
    data = CSVData(args.file)
    id3 = ID3(data)
    tree = id3.run()
    tree.show()
    measure = PerformanceTest(tree, data, args.verbose)
    measure.print_statistics()
    measure.print_test()


if __name__ == '__main__':
    main()
