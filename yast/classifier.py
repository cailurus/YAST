#!/usr/bin/env python

import sys
from os import path
import os
import shutil
import operator
from converter import Text2svmConverter
from .learner import LIBLINEAR_HOME, LearnerModel

VERSION = 0

__all__ = ['VERSION', 'PredictionResult', 'TextModel', 'LIBLINEAR_HOME']

class PredictionResult(object):
    _VERSION = 1

    def __init__(self, text_src = None, model_id = None, true_y = None, predicted_y = None, decvals = None, svm_file = None, labels = None, extra_svm_files = []):
        self.text_src = text_src
        self.model_id = model_id
        self.true_y = true_y
        self.predicted_y = predicted_y
        self.decvals = decvals
        self.svm_file = svm_file
        self.labels = labels

        self.extra_svm_files = extra_svm_files

    def get_accuracy(self):
        if not self.analyzable():
            raise RuntimeError('Only analyzable PredictionResult has accuracy.')

        true_y = self.true_y
        predicted_y = self.predicted_y
        l = len(self.true_y)
        return sum([true_y[i] == predicted_y[i] for i in range(l)]) / float(l)

    def save(self, file_name, analyzable = False, fmt = '.16g'):
        fout = open(file_name, 'w')

        if type(self.predicted_y) is not list:
            raise ValueError('PredictionResult is only writable for single instance.')

        if analyzable is True and not self.analyzable():
            raise ValueError('PredictionResult is not analyzable if text_src or model_id is not given.')

        fout.write('version: {0}\n'.format(self._VERSION))
        fout.write('analyzable: {0}\n'.format(int(analyzable)))
        if analyzable:
            fout.write('text-src: {0}\n'.format(self.text_src))
            fout.write('extra-files:\t{0}\n'.format('\t'.join(self.extra_svm_files)))
            fout.write('model-id: {0}\n'.format(self.model_id))
        fout.write('\n')

        if not analyzable:
            for y in self.predicted_y:
                fout.write(y + "\n")
            fout.close()
            return

        fmt = '\t{{0:{0}}}'.format(fmt)
        for i in range(len(self.predicted_y)):
            fout.write("{py}\t{y}".format(py = self.predicted_y[i], y = self.true_y[i]))
            for v in self.decvals[i]:
                fout.write(fmt.format(v))
            fout.write('\n')

        fout.close()

    def get_label_dict(self):
        return dict(zip(self.labels, self.decvals))

    def get_tags(self, num):
        dec_dict = self.get_label_dict()
        labels_sort = [kv[0] for kv in sorted(dec_dict.items(), key=operator.itemgetter(1))[::-1][:num]]
        labels_sort_dict = {}
        for tag in labels_sort:
            labels_sort_dict[tag] = dec_dict[tag]
        return labels_sort_dict


    def load(self, file_name):
        self.text_src = None
        self.model_id = None
        self.true_y = None
        self.predicted_y = None
        self.decvals = None
        self.svm_file = None
        self.labels = None
        self.extra_svm_files = []

        fin = open(file_name, 'r')

        analyzable = False

        for line in fin:
            line = line.strip()
            if line == '':
                break

            if line.startswith('version: '):
                v = float(line.split(None, 1)[1])
                if self._VERSION < v:
                    raise Exception("This result is not supported in this version. Please update package.")
            elif line.startswith('analyzable: '):
                analyzable = int(line.split(None, 1)[1])
            elif line.startswith('text-src: '):
                self.text_src = line.split(None, 1)[1]
            elif line.startswith('extra-files:'):
                self.extra_svm_files = line.split('\t')[1:]
            elif line.startswith('model-id: '):
                self.model_id = line.split(None, 1)[1]
            else:
                raise ValueError("Unexpected argument " + str(line))

        if not analyzable:
            self.predicted_y = [line.strip() for line in fin]
            fin.close()
            return

        self.predicted_y = []
        self.true_y = []
        self.decvals = []

        for line in fin:
            py, y, line = line.split('\t', 2)

            self.predicted_y += [py]
            self.true_y += [y]
            self.decvals += [list(map(lambda x: float(x), line.split()))]

        fin.close()

    def analyzable(self):
        if None in [self.decvals, self.predicted_y, self.true_y, self.text_src, self.model_id]:
            return False

        if type(self.decvals) is not list or type(self.predicted_y) is not list or type(self.true_y) is not list:
            return False

        return True

    def __str__(self):
        if self.analyzable():
            return 'analyzable result (size = {0}): (data: {1}, accuracy: {2})'.format(len(self.predicted_y), self.text_src, self.get_accuracy())
        else:
            return 'unanalyzable results: {0}, '.format(self.predicted_y)

class TextModel(object):
    def __init__(self, arg1 = None, arg2 = None):
        if arg2 is None and isinstance(arg1, str):
            self.load(arg1)
            return

        if arg1 is None or isinstance(arg1, Text2svmConverter):
            self.text_converter = arg1
        if arg2 is None or isinstance(arg2, LearnerModel):
            self.svm_model = arg2

        # generate an id for TextModel
        from hashlib import sha1
        from time import time

        self._hashcode = sha1('{0} {1:.16g}'.format(id(self), time()).encode('ASCII')).hexdigest()

    def __str__(self):
        return 'TextModel instance ({0}, {1})'.format(self.text_converter, self.svm_model)

    def get_labels(self):
        return [self.text_converter.getClassName(k) for k in self.svm_model.get_labels()]

    def load(self, model_name):
        try:
            fin = open(model_name + '/id', 'r')
            self._hashcode = fin.readline().strip()
            fin.close()
        # except IOError as ex:
        except IOError:
            raise ValueError("The given model is invalid.")

        self.text_converter = Text2svmConverter().load(model_name + '/converter')
        self.svm_model = LearnerModel(model_name + '/learner')


    def save(self, model_name, force=False):
        if self.svm_model == None:
            raise Exception('This model can not be saved because svm model is not given.')

        if path.exists(model_name) and force:
            shutil.rmtree(model_name)

        try:
            os.mkdir(model_name)
        except OSError as e:
            raise OSError(e, 'Please use force option to overwrite the existing files.')

        self.text_converter.save(model_name + '/converter')
        self.svm_model.save(model_name + '/learner', force)

        fout = open(model_name + '/id', 'w')
        fout.write(self._hashcode)
        fout.close()


    def get_weight(self, xi, labels = None, extra_svm_feats = []):
        if self.svm_model == None:
            raise Exception('This model is not usable because svm model is not given.')

        if labels is None:
            labels = self.get_labels()
        elif isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            raise TypeError('labels should be a list of classes.')


        if isinstance(xi, str):
            xi = self.text_converter.toSVM(xi, extra_svm_feats = extra_svm_feats)
        elif extra_svm_feats != 0:
            sys.stderr.write('Warning: extra_svm_feats ignored')

        if isinstance(xi, list):
            xi = dict(filter(lambda i: i[1] is not 0, enumerate(xi, 1)))
        elif not isinstance(xi, dict):
            raise TypeError('xi should be a sentence or a LIBSVM instance.')

        numLabels = [self.text_converter.getClassIdx(k) for k in labels]
        not_found_labels = [labels[k]
                for k in filter(lambda lb: numLabels[lb] is None, range(len(numLabels)))]
        if len(not_found_labels) > 0:
            raise KeyError('The following labels do not exist: ' + ','.join(not_found_labels))

        weights = [ [self.svm_model.get_weight(j, k) for k in numLabels]
                for j in xi]

        features = [self.text_converter.get_fidx2tok(j) for j in xi]

        return (features, weights, labels)
