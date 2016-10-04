#!/usr/bin/env python
# encoding: utf-8

import sys
import stat
import os
from grid import find_parameters
from .learner import train, predict, predict_one
from classifier import TextModel, PredictionResult
from converter import convert_text,Text2svmConverter


def train_converted_text(svm_file, text_converter, grid_arguments = '0', feature_arguments = '', train_arguments = ''):
    '''
    return a TextModel
    '''
    train_arguments = '-s 4 ' + train_arguments

    if grid_arguments != '0' and grid_arguments != 0:
        grid_multi_path = os.path.dirname(__file__)+'/learner/learner_impl.py'
        default_grid_arguments = '-svmtrain {0} -log2g null -log2c -6,6,2 '.format(os.path.dirname(__file__) + '/learner/learner_impl.py')
        # default_grid_arguments = '-svmtrain /home/pio/yast/yast/learner/learner_impl.py -log2g null -log2c -6,6,2 '#.format(path.dirname(__file__) + '/learner/learner_impl.py')
        if grid_arguments == '1' or grid_arguments == 1:
            try:
                os.chmod(grid_multi_path, stat.S_IRWXU) # Execute by owner.
            except OSError:
                raise OSError("Grid search needs sudo")
            grid_arguments = default_grid_arguments
        else:
            grid_arguments = default_grid_arguments + grid_arguments
        # print "ggg" , svm_file, "||", grid_arguments , ' || ',  train_arguments , '|| ',  feature_arguments
        parameters = find_parameters(svm_file, grid_arguments + ' ' + train_arguments + ' ' + feature_arguments)[1]
        train_arguments = train_arguments + ' -c ' + str(parameters["c"])

    m = train(svm_file, feature_arguments, train_arguments)

    return TextModel(text_converter, m)

def train_text(text_src, svm_file = None, converter_arguments = '', grid_arguments = '0', feature_arguments = '', train_arguments = '', extra_svm_files = []):
    """
    return (TextModel, LIBSVM)
    """

    # name = path.split(text_src)[1]
    # svm_file = svm_file or name + '.svm'

    text_converter = Text2svmConverter(converter_arguments)
    convert_text(text_src, text_converter, svm_file)
    text_converter.merge_svm_files(svm_file, extra_svm_files)

    m = train_converted_text(svm_file, text_converter, grid_arguments, feature_arguments, train_arguments)
    return (m, svm_file)

def predict_text(text_src, text_model, svm_file=None, predict_arguments='', extra_svm_files = []):
    """
    return PredictionResult with analyzable feature
    """
    # name = path.split(text_src)[1]
    # svm_file = svm_file or name + '.svm'

    text_converter = text_model.text_converter
    convert_text(text_src, text_converter, svm_file)
    text_converter.merge_svm_files(svm_file, extra_svm_files)

    predicted_y, acc, dec_vals, true_y = predict(svm_file, text_model.svm_model, predict_arguments)

    predicted_y = [text_model.text_converter.getClassName(int(y)) for y in predicted_y]
    true_y = [text_model.text_converter.getClassName(int(y)) for y in true_y]

    return PredictionResult(text_src, text_model._hashcode, true_y, predicted_y, dec_vals, svm_file, text_model.get_labels(), extra_svm_files)

def predict_single_text(text, text_model, predict_arguments = '', extra_svm_feats = []):
    '''
    return an unanalyzable obj
    '''
    if not isinstance(text_model, TextModel):
        raise TypeError('argument 1 should be TextModel')

    if text_model.svm_model == None:
        raise Exception('This model is not usable because svm model is not given')

    if isinstance(text, str):
        text = text_model.text_converter.toSVM(text, extra_svm_feats = extra_svm_feats)
    elif isinstance(text, (list, dict)):
        if extra_svm_feats:
            sys.stderr.write('Warning: extra_svm_feats ignored')
    else:
        raise TypeError('The argument should be plain text or LIBSVM-format data.')

    y, dec = predict_one(text, text_model.svm_model)

    y = text_model.text_converter.getClassName(int(y))
    labels = [text_model.text_converter.getClassName(k) for k in text_model.svm_model.label[:text_model.svm_model.nr_class]]

    return PredictionResult(predicted_y = y, decvals = dec[:text_model.svm_model.nr_class], labels = labels)
