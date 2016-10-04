#!/usr/bin/env python
# encoding: utf-8

from os import path
from classifier import TextModel
from do_classify import train_text, predict_text, predict_single_text
from analyzer import Analyzer, InstanceSet
from anlys_selector import with_labels
from config import configs

class YastException(Exception):
    pass


class YastModelNotTrainException(YastException):
    def __init__(self):
        self.message = 'Text model has not been trained.'


class Yast(object):
    def __init__(self, name, configs=configs):
        self.name = name
        self.model = None
        self.model_path = '%s.model' % self.name
        self.classifier = None
        self.train_svm_file = None
        self.test_result = None
        self.configs = self.convert_configs(configs)
        self.stopword = None

    def get_load_status(self):
        return self.model is not None and isinstance(self.model, TextModel)

    def load_model(self):
        self.model = TextModel(self.model_path)

    def load_stopword(self, sfile):
        if not isinstance(sfile, str):
            raise TypeError("Should set the path to stopwords")

    def convert_configs(self, custom_configs=None):
        default_configs = {
            'converter_arguments': '-stopword 0 -stemming 0 -feature 1',
            'grid_arguments': '0',
            'feature_arguments': '3',
            'liblinear_arguments': '', # default is -s 4
        }
        if 'grid' in custom_configs:
            default_configs['grid_arguments'] = custom_configs['grid']
        if 'stopword' in custom_configs:
            default_configs['converter_arguments'] = '-stopword ' + str(custom_configs['stopword']) + ' -stemming 0 -feature 1'

        if 'feature' in custom_configs:
            if custom_configs['feature'] == 0:
                default_configs['feature_arguments'] = '-D 1'
            elif custom_configs['feature'] == 1:
                default_configs['feature_arguments'] =  '-D 0'
            elif custom_configs['feature'] == 2:
                default_configs['feature_arguments'] =  '-D 0 -T 1'
            elif custom_configs['feature'] == 3:
                default_configs['feature_arguments'] =  '-D 0 -T 1 -I 1'
        if 'classifier' in custom_configs:
            if custom_configs['classifier'] == 0:
                default_configs['liblinear_arguments'] = '-s 4'
            elif custom_configs['classifier'] == 1:
                default_configs['liblinear_arguments'] = '-s 3'
            elif custom_configs['classifier'] == 2:
                default_configs['liblinear_arguments'] =  '-s 1'
            elif custom_configs['classifier'] == 3:
                default_configs['liblinear_arguments'] =  '-s 7'
        return default_configs

    def train(self, train_src, delimiter='\t'):
        converter_arguments = self.configs['converter_arguments']
        grid_arguments      = self.configs['grid_arguments']
        feature_arguments   = self.configs['feature_arguments']
        liblinear_arguments = self.configs['liblinear_arguments'] # default is -s 4
        force               = True #self.onfigs['force']
        extra_svm_files     = []
        self.train_svm_file = '%s_train.svm' % self.name
        m, svm_file = train_text(train_src, svm_file=self.train_svm_file, converter_arguments=converter_arguments, grid_arguments=grid_arguments, feature_arguments=feature_arguments, train_arguments=liblinear_arguments, extra_svm_files = extra_svm_files)
        self.model = m
        m.save(self.model_path, force)

    def test(self, test_src):
        liblinear_arguments = self.configs['liblinear_arguments']
        analyzable          = True
        extra_svm_files     = []
        test_svm_file = '%s_test.svm' % self.name

        predict_result = predict_text(test_src, self.model, svm_file=test_svm_file, predict_arguments=liblinear_arguments, extra_svm_files = extra_svm_files)

        print("Accuracy = {0:.4f}% ({1}/{2})".format(
            predict_result.get_accuracy()*100,
            sum(ty == py for ty, py in zip(predict_result.true_y, predict_result.predicted_y)),
            len(predict_result.true_y)))

        self.test_result = '%s_test_data_svm.anlyz' % self.name
        predict_result.save(self.test_result, analyzable)

    def predict_single(self, single_text):
        liblinear_arguments = ''
        extra_svm_files     = []
        predict_result = predict_single_text(single_text, self.model, predict_arguments = liblinear_arguments, extra_svm_feats = extra_svm_files)
        return predict_result

    def analyze(self, stext):
        insts = InstanceSet(self.test_result)
        insts.load_text()
        analyzer = Analyzer(self.model_path)
        analyzer.analyze_single(stext)


# def predict_single_text(text, text_model, predict_arguments = '', extra_svm_feats = []):


#        text_converter = GroceryTextConverter(custom_tokenize=self.custom_tokenize)
#        self.train_svm_file = '%s_train.svm' % self.name
#        text_converter.convert_text(train_src, output=self.train_svm_file, delimiter=delimiter)
#        # default parameter
#        model = train(self.train_svm_file, '', '-s 4')
#        self.model = GroceryTextModel(text_converter, model)
#        return self

#    def predict(self, single_text):
#        if not self.get_load_status():
#            raise GroceryNotTrainException()
#        return self.model.predict_text(single_text)
#
#    def test(self, text_src, delimiter='\t'):
#        if not self.get_load_status():
#            raise GroceryNotTrainException()
#        return GroceryTest(self.model).test(text_src, delimiter)
#
#    def save(self):
#        if not self.get_load_status():
#            raise GroceryNotTrainException()
#        self.model.save(self.name, force=True)
#
#    def load(self):
#        self.model = GroceryTextModel(self.custom_tokenize)
#        self.model.load(self.name)
#
#    def __del__(self):
#        if self.train_svm_file and os.path.exists(self.train_svm_file):
#            os.remove(self.train_svm_file)
