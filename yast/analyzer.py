#!/usr/bin/env python
# encoding: utf-8

import sys, os
from classifier import TextModel, PredictionResult
from do_classify import predict_single_text
from anlys_utils import draw_table, write

__all__ = ['TextInstance', 'InstanceSet', 'Analyzer']

if sys.version_info[0] >= 3:
    xrange = range
    izip = zip
else:
    from itertools import izip

class TextInstance:
    def __init__(self, idx, true_y = '', predicted_y = '', text = '', extra_svm_feats = [], decvals = None):
        self.idx = idx #: Instance index in the text source.
        self.true_y = true_y
        self.predicted_y = predicted_y
        self.text = text
        self.extra_svm_feats = extra_svm_feats
        self.decvals = decvals

    def __str__(self):
        string = '''text = {text}
                    true label = {true_y}
                    predicted label = {predicted_y}
                 '''.format(text = self.text, true_y = self.true_y, predicted_y = self.predicted_y)
        if self.extra_svm_feats:
        	string += 'extra svm features = {extra}\n'.format(extra = self.extra_svm_feats)
        return string

class InstanceSet:
    '''
    group of TextInstance
    '''
    def __init__(self, rst_src = None, text_src = None):
        self.insts = None
        self.correct = None
        self.filepath = None
        self.extra_svm_files = []
        self.true_labels = None
        self.predict_labels = None
        self.quantity = None
        self.selectors = []
        if rst_src is not None:
            self._load(rst_src, text_src)

    def __iter__(self):
        return iter(self.insts)

    def __getitem__(self, idx):
        return self.insts[idx]

    def select(self, *sel_funcs):
        '''
        select with featured labels
        '''
        ### How to link to the section??
        insts = self.insts
        selectors = self.selectors[:]
        for sel_func in sel_funcs:
            insts = sel_func(insts)
            selectors.append(sel_func._libshorttext_msg or '')
        #if not insts:
        #	raise Exception("No instance selected.")
        sel_insts = InstanceSet()
        sel_insts.filepath = self.filepath
        sel_insts.extra_svm_files = self.extra_svm_files
        sel_insts.selectors = selectors
        sel_insts.insts = insts
        return sel_insts

    def load_text(self):
        EMPTY_MESSAGE = '**None**'
        sorted_insts = sorted(self.insts, key = lambda inst: inst.idx)
        i = 0
        for idx, lines in enumerate(izip(*([open(self.filepath, 'r')] + [open(f, 'r') for f in self.extra_svm_files]))):
            line = lines[0]
            extra_svm_feats = lines[1:]
            nr_extra_svm_feats = len(extra_svm_feats)
            if idx > sorted_insts[-1].idx:
                break
            if idx == sorted_insts[i].idx:
                try:
                    sorted_insts[i].text = line.split('\t',1)[1].strip()
                except:
                    sorted_insts[i].text = EMPTY_MESSAGE

                sorted_insts[i].extra_svm_feats = [None] * nr_extra_svm_feats
                for j, extra_svm_feat in enumerate(extra_svm_feats):
                    try:
                        sorted_insts[i].extra_svm_feats[j] = dict(map(lambda t: (int(t[0]), float(t[1])), [feat.split(':') for feat in extra_svm_feat.split(None, 1)[1].split()]))
                    except:
                        sorted_insts[i].extra_svm_feats[j] = EMPTY_MESSAGE
                i += 1

    def _load(self, src, text_src):
        if isinstance(src, PredictionResult):
            pass
        elif isinstance(src, str):
            result = PredictionResult()
            result.load(src)
        else:
            raise Exception('"result" should be PredictionResult or string.')

        if not result.analyzable():
            raise ValueError('The given result is not analyzable.')

        # +++ Need to move to another place.
        #if self.model._hashcode != result.model_id:
        #	sys.stderr.write('Warning: model ID is different from that in the predicted result. Do you use a different model to analyze?\n')

        if text_src is None:
            self.filepath = result.text_src
        else:
            self.filepath = text_src
        self.extra_svm_files = result.extra_svm_files
        predicted_y = result.predicted_y
        self.acc = result.get_accuracy()
        decvals = result.decvals
        true_y = result.true_y

        self.insts, self.true_labels, self.predict_labels = [], set(), set()
        for idx in range(len(true_y)):
        	self.insts += [TextInstance(idx, true_y = true_y[idx], predicted_y = predicted_y[idx], decvals = list(decvals[idx]))]
        	self.true_labels.add(true_y[idx])
        	self.predict_labels.add(predicted_y[idx])

class Analyzer:
    '''
    init with a TextModel
    '''

    def __init__(self, model = None):
        self.labels = None
        self.model = None
        if model is not None:
            self.load_model(model)

    def load_model(self, model):
        '''
        load with TextModel path or obj
        '''

        if isinstance(model, TextModel):
            self.model = model
        elif isinstance(model, str):
            self.model = TextModel()
            self.model.load(model)
        else:
            raise Exception('"model" should be TextModel or string.')
        self.labels = self.model.get_labels()

    def analyze_single(self, target, amount = 5, output = None, extra_svm_feats = []):
        '''
        return analyze result of a query
        results will be stored into disk if output is specified
        '''
        if self.model is None:
            raise Exception('Model not loaded.')
        if isinstance(target,str):
            text = target
            true_y = None
            result = predict_single_text(text, self.model, extra_svm_feats = extra_svm_feats)
            decvals = result.decvals
        elif isinstance(target,TextInstance):
            if target.text is None:
                raise Exception('Please load texts first.')
            text, extra_svm_feats, true_y = target.text, target.extra_svm_feats, target.true_y
            decvals = target.decvals
        if isinstance(output, str):
            output = open(output, 'w')

        features, weights, labels = self.model.get_weight(text, extra_svm_feats = extra_svm_feats)
        nr_labels = len(labels)
        nr_feats = len(features)
        if not features or not weights:
            raise Exception('Invalid instance.')
        features = [' '.join(feature) for feature in features]
        features += ['**decval**']
        weights_table = [[0]*nr_labels]*(nr_feats+1)
        sorted_idx = sorted(xrange(nr_labels), key=lambda i:decvals[i], reverse=True)
        labels = [labels[idx] for idx in sorted_idx]


        for feat in xrange(nr_feats):
            formatter = lambda idx: '{0:.3e}'.format(weights[feat][idx])
            weights_table[feat] = [formatter(idx) for idx in sorted_idx]
        weights_table[-1] = ['{0:.3e}'.format(decvals[idx]) for idx in sorted_idx]

        if amount != 0:
            labels = labels[:amount]
        draw_table(features, labels, weights_table, output)
        if true_y is not None:
            print('True label: {0}'.format(true_y))

    def _calculate_info(self, pred_insts):
        pred_insts.quantity = len(pred_insts.insts)
        pred_insts.true_labels, pred_insts.predict_labels, pred_insts.correct = set(), set(), 0
        for inst in pred_insts.insts:
            pred_insts.true_labels.add(inst.true_y)
            pred_insts.predict_labels.add(inst.predicted_y)
            if inst.true_y == inst.predicted_y:
                pred_insts.correct += 1

    def info(self, pred_insts, output = None):
        '''
        print insts info
        '''
        if isinstance(output, str):
            output = open(output, 'w')
        if pred_insts.quantity is None:
            self._calculate_info(pred_insts)
        acc = float(pred_insts.correct)/pred_insts.quantity

        string = '''Number of instances: {quantity}
                    Accuracy: {acc} ({correct}/{quantity})
                    True labels: {true_y}
                    Predicted labels: {predicted_y}
                    Text source: {text_src}
                    Selectors: \n-> {selectors}
                 '''.format(quantity = pred_insts.quantity, correct = pred_insts.correct,\
                            acc = acc, true_y = '"'+'"  "'.join(pred_insts.true_labels)+'"',\
                            predicted_y = '"'+'"  "'.join(pred_insts.predict_labels)+'"',\
                            text_src = os.path.abspath(pred_insts.filepath),\
                            selectors = '\n-> '.join(pred_insts.selectors))

        write(string, output)

    def gen_confusion_table(self, pred_insts, output = None):
        '''
        draw a debug table to analyze
        '''
        if isinstance(output, str):
            output = open(output, 'w')
        if pred_insts.quantity is None:
            self._calculate_info(pred_insts)
        labels = pred_insts.true_labels.union(pred_insts.predict_labels)
        #columns = rows

        invalid_labels = []
        for label in labels:
            if label not in pred_insts.true_labels and label not in pred_insts.predict_labels:
                invalid_labels.append(label)
        if invalid_labels:
            invalid_labels = ' '.join(invalid_labels)
            raise Exception('Labels {0} are invalid.'.format(invalid_labels))

        labels_dic = dict(zip(labels, xrange(len(labels))))
        confusion_table = [[0 for i in range(len(labels_dic))] for j in range(len(labels_dic))]
        for inst in pred_insts.insts:
            if inst.true_y in labels_dic and inst.predicted_y in labels_dic:
                confusion_table[labels_dic[inst.true_y]][labels_dic[inst.predicted_y]] += 1
        for idx_row, row in enumerate(confusion_table):
            for idx_col, col in enumerate(row):
                confusion_table[idx_row][idx_col] = str(confusion_table[idx_row][idx_col])

        draw_table(labels, labels, confusion_table, output)

        if output:
            output.close()
