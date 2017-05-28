#!/usr/bin/env python
# encoding:utf-8

__all__ = ["TextPreprocessor", "FeatureGenerator", "ClassMapping", "Text2svmConverter", "convert_text"]

import sys, os
import unicodedata, re
import utils
from collections import defaultdict
import jieba
from text_cleaner.processor.common import SYMBOLS_AND_PUNCTUATION_EXTENSION, SYMBOLS_AND_PUNCTUATION, GENERAL_PUNCTUATION
from text_cleaner.processor.chinese import CHINESE_SYMBOLS_AND_PUNCTUATION
from text_cleaner import remove, keep

if sys.version_info[0] >= 3:
    xrange = range
    import pickle as cPickle
    izip = zip
    def unicode(string, setting):
        return string
else :
    import cPickle
    # from itertools import izip

# import porter stemmer
# from .stemmer import porter


class TextPreprocessor(object):
    def __init__(self, option='-stemming 0 -stopword 0', readonly=False):
        # jieba.load_userdict('./sample_data/stopwords.txt')
        self._option  = option
        self._readonly=readonly
        self.tok2idx = {'>>dummy<<':0}
        self.idx2tok = None
        opts = self.parse_option(option)
        #: The function used to stem tokens.
        #:
        #: Refer to :ref:`CustomizedPreprocessing`.
        self.stemmer = opts[0]
        #: The function used to remove stop words.
        #:
        #: Refer to :ref:`CustomizedPreprocessing`.
        self.stopword_remover = opts[1]
        #: The function used to tokenize texts into a :class:`list` of tokens.
        #:
        #: Refer to :ref:`CustomizedPreprocessing`.
        self.tokenizer = self.default_tokenizer

    def parse_option(self, option):
        option = option.strip().split()
        stoplist, tokstemmer = set(), lambda x: x
        i = 0
        while i < len(option):
            if option[i][0] != '-': break
            if option[i] == '-stopword':
                if int(option[i+1]) != 0:
                    print "stop did!"
                    stoplist = self.default_stoplist()
            elif option[i] == '-stemming':
                if int(option[i+1]) != 0:
                    tokstemmer = porter.stem
            i+=2
        stoplist = set(tokstemmer(x) for x in stoplist)
        stemmer = lambda text: map(tokstemmer, text)
        stopword_remover = lambda text: filter(lambda tok: tok not in stoplist, text)
        return stemmer, stopword_remover

    def get_idx2tok(self, idx):
        if not self.idx2tok:
            self.idx2tok = utils.dict2list(self.tok2idx)
        return self.idx2tok[idx]

    def save(self, dest_file):
        self.idx2tok = utils.dict2list(self.tok2idx)
        config = {'option':self._option,'idx2tok':self.idx2tok}
        cPickle.dump(config, open(dest_file,'wb'), -1)

    # by default, mapping file will be not updated when we load the file
    def load(self, src_file, readonly=True):
        config = cPickle.load(open(src_file,'rb'))
        self._readonly = readonly
        self._option = config['option']
        self.idx2tok = config['idx2tok']
        self.tok2idx = utils.list2dict(config['idx2tok'])
        self.stemmer, self.stopword_remover = self.parse_option(config['option'])
        self.tokenizer = self.default_tokenizer
        return self

    @staticmethod
    def default_stoplist_en():
        # This function only parses the default stop word list file.
        # *src* should not be an argument.
        src = ""
        if not src:
            src = '{0}/stop-words/stoplist-nsp.regex'.format(os.path.dirname(os.path.abspath(__file__)))
        srcfile = open(src)
        stoplist = set(map(chr, range(ord('a'),ord('z')+1)))
        srcfile.readline()
        srcfile.readline()
        for line in srcfile:
            stoplist.add(line[5:-4].lower().replace(']',''))
        return stoplist

    @staticmethod
    def default_stoplist():
        jieba.load_userdict('./sample_data/stopwords.txt')
        stoplist = []
        with open ('./sample_data/stopwords.txt') as f:
            infos = f.readlines()
            stoplist = [i.strip().decode('utf-8') for i in infos]
            stoplist.append(u" ")

        return stoplist

    @staticmethod
    def default_tokenizer_en(text):
        def foo(c):
            if ord(c)>127: return ''
            if c.isdigit() or c.isalpha(): return c
            else : return ' '

        text = unicodedata.normalize('NFD', unicode(text, 'utf-8')).lower()
        text = ''.join(map(foo,text))
        text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)
        text = re.sub(r'([0-9])([a-z])', r'\1 \2', text)
        text = re.sub(r'\s+', r' ', text)
        return text.strip().split()

    @staticmethod
    def default_tokenizer(text):

        return jieba.cut_for_search(text)

    def preprocess(self, text):
        # text = remove(text, [CHINESE_SYMBOLS_AND_PUNCTUATION, SYMBOLS_AND_PUNCTUATION_EXTENSION, SYMBOLS_AND_PUNCTUATION, GENERAL_PUNCTUATION])
        # text = self.punc_remover(text)
        text = self.tokenizer(text)
        # text = self.stemmer(text)
        text = self.stopword_remover(text)

        ret = []
        for i,tok in enumerate(text):
            if tok not in self.tok2idx:
                if self._readonly: continue
                self.tok2idx[tok] = len(self.tok2idx)
                self.idx2tok = None
            ret += [self.tok2idx[tok]]
        return ret


class FeatureGenerator(object):
    """
    generate uni-gram or bi-gram features.
    """

    def __init__(self, option='-feature 1', readonly=False):
        self._option = option
        self._readonly = readonly
        self.ngram2fidx = {'>>dummy<<':0}
        self.fidx2ngram=None
        self.feat_gen = self.parse_option(option)

    def parse_option(self, option):
        option = option.strip().split()
        feat_gen = self.bigram
        i = 0
        while i < len(option):
            if option[i][0] != '-': break
            if option[i] == '-feature':
                if int(option[i+1]) == 0:
                    feat_gen = self.unigram
            i+=2
        return feat_gen

    def get_fidx2ngram(self, fidx):
        if not self.fidx2ngram:
            self.fidx2ngram = utils.dict2list(self.ngram2fidx)
        return self.fidx2ngram[fidx]

    def save(self, dest_file):
        self.fidx2ngram = utils.dict2list(self.ngram2fidx)
        config = {'option':self._option,'fidx2ngram':self.fidx2ngram}
        cPickle.dump(config, open(dest_file,'wb'), -1)

    # by default, mapping file will be not updated when we load the file
    def load(self, src_file, readonly=True):
        config = cPickle.load(open(src_file,'rb'))
        self._option = config['option']
        self.fidx2ngram = config['fidx2ngram']
        self.ngram2fidx = utils.list2dict(config['fidx2ngram'])
        self._readonly=readonly
        self.feat_gen = self.parse_option(config['option'])
        return self

    def toSVM(self, text):
        return self.feat_gen(text)
        #return ''.join(' %d:%d' %(f, feat[f]) for f in sorted(feat))

    def unigram(self, text):
        feat = defaultdict(int)
        NG = self.ngram2fidx
        for x in text:
            if (x,) not in NG:
                if self._readonly: continue
                NG[x,] = len(NG)
                self.fidx2ngram = None
            feat[NG[x,]]+=1
        return feat

    def bigram(self, text):
        feat = self.unigram(text)
        NG = self.ngram2fidx
        for x,y in zip(text[:-1], text[1:]):
            if (x,y) not in NG:
                if self._readonly: continue
                NG[x,y] = len(NG)
                self.fidx2ngram = None
            feat[NG[x,y]]+=1
        return feat


class ClassMapping(object):
    def __init__(self, option='', readonly=False):
        self._option = option
        self._readonly = readonly
        self.class2idx = {}
        self.idx2class = None

    def save(self, dest_file):
        self.idx2class = utils.dict2list(self.class2idx)
        config = {'option':self._option,'idx2class':self.idx2class}
        cPickle.dump(config, open(dest_file,'wb'), -1)

    # by default, mapping file will be not updated when we load the file
    def load(self, src_file, readonly=True):
        config = cPickle.load(open(src_file,'rb'))
        self._readonly = readonly
        self._option = config['option']
        self.idx2class = config['idx2class']
        self.class2idx = utils.list2dict(config['idx2class'])
        return self


    def toIdx(self, class_name):
        if class_name in self.class2idx:
            return self.class2idx[class_name]
        elif self._readonly:
            return None

        m = len(self.class2idx)
        self.class2idx[class_name] = m
        self.idx2class = None
        return m

    def toClassName(self, idx):
        if self.idx2class is None:
            self.idx2class = utils.dict2list(self.class2idx)
        if idx == -1:
            return "**not in training**"
        if idx >= len(self.idx2class):
            raise KeyError('class idx ({0}) should be less than the number of classes ({0}).'.format(idx, len(self.idx2class)))
        return self.idx2class[idx]

    def rename(self, old_label, new_label):
        if not isinstance(new_label, str):
            raise TypeError("new_label should be a str")

        if isinstance(old_label, int):
            old_label = self.toClassName(old_label)
        if isinstance(old_label, str):
            if old_label not in self.class2idx:
                raise ValueError('class {0} does not exist'.format(old_label))
        else:
            raise TypeError("old label should be int (index) or str (name)")

        if new_label in self.class2idx:
            raise ValueError('class {0} already exists'.format(new_label))

        self.class2idx[new_label] = self.class2idx.pop(old_label)
        self.idx2class = None


class Text2svmConverter(object):
    '''
    TextPreprocessor -> FeatureGenerator -> ClassMapping

    '''
    def __init__(self, option="", readonly=False):
        self._option = option
        self._readonly = readonly
        self._extra_nr_feats = []
        self._extra_file_ids = []
        text_prep_opt, feat_gen_opt, class_map_opt = self._parse_option(option)
        #: The :class:`TextPreprocessor` instance.
        self.text_prep = TextPreprocessor(text_prep_opt, readonly)
        #: The :class:`FeatureGenerator` instance.
        self.feat_gen = FeatureGenerator(feat_gen_opt, readonly)
        #: The :class:`ClassMapping` instance.
        self.class_map = ClassMapping(class_map_opt, readonly)

    def _parse_option(self, option):
        text_prep_opt, feat_gen_opt, class_map_opt = '', '', ''
        option = option.strip().split()
        i = 0
        while i < len(option):
            if i+1 >= len(option):
                raise ValueError("{0} cannot be the last option.".format(option[i]))

            if type(option[i+1]) is not int and not option[i+1].isdigit():
                raise ValueError("Invalid option {0} {1}.".format(option[i], option[i+1]))
            if option[i] in ['-stopword', '-stemming']:
                text_prep_opt = ' '.join([text_prep_opt, option[i], option[i+1]])
            elif option[i] in ['-feature']:
                feat_gen_opt = ' '.join([feat_gen_opt, option[i], option[i+1]])
            else:
                raise ValueError("Invalid option {0}.".format(option[i]))
            i+=2
        return text_prep_opt, feat_gen_opt, class_map_opt

    def merge_svm_files(self, svm_file, extra_svm_files):
        if not isinstance(extra_svm_files, (tuple, list)):
            raise TypeError('extra_svm_files should be a tuple or a list')

        nr_files = len(extra_svm_files)

        if self._readonly: # test
            if len(self._extra_file_ids) != nr_files:
                raise ValueError('wrong number of extra svm files ({0} expected)'.format(len(self._extra_file_ids)))
            if nr_files == 0: return

            utils.merge_files([svm_file] + extra_svm_files, self._extra_nr_feats, False, svm_file)

        else: # train
            if nr_files == 0: return

            self._extra_file_ids = [os.path.basename(f) for f in extra_svm_files]
            self._extra_nr_feats = [0] * (nr_files + 1)
            utils.merge_files([svm_file] + extra_svm_files, self._extra_nr_feats, True, svm_file)


    def save(self, dest_dir):
        """
        Save the model to a directory.
        """

        config = {'text_prep':'text_prep.config.pickle',
                  'feat_gen':'feat_gen.config.pickle',
                  'class_map':'class_map.config.pickle',
                  'extra_nr_feats': 'extra_nr_feats.pickle',
                  'extra_file_ids': 'extra_file_ids.pickle'}
        if not os.path.exists(dest_dir): os.mkdir(dest_dir)
        self.text_prep.save(os.path.join(dest_dir,config['text_prep']))
        self.feat_gen.save(os.path.join(dest_dir,config['feat_gen']))
        self.class_map.save(os.path.join(dest_dir,config['class_map']))

        cPickle.dump(self._extra_nr_feats, open(os.path.join(dest_dir, config['extra_nr_feats']), 'wb'), -1)
        cPickle.dump(self._extra_file_ids, open(os.path.join(dest_dir, config['extra_file_ids']), 'wb'), -1)

    def load(self, src_dir, readonly=True):
        """
        Load the model from a directory.
        """

        self._readonly = readonly

        config = {'text_prep':'text_prep.config.pickle',
                  'feat_gen':'feat_gen.config.pickle',
                  'class_map':'class_map.config.pickle',
                  'extra_nr_feats': 'extra_nr_feats.pickle',
                  'extra_file_ids': 'extra_file_ids.pickle'}
        self.text_prep.load(os.path.join(src_dir,config['text_prep']),readonly)
        self.feat_gen.load(os.path.join(src_dir,config['feat_gen']),readonly)
        self.class_map.load(os.path.join(src_dir,config['class_map']),readonly)

        self._extra_nr_feats = cPickle.load(open(os.path.join(src_dir, config['extra_nr_feats']), 'rb'))
        self._extra_file_ids = cPickle.load(open(os.path.join(src_dir, config['extra_file_ids']), 'rb'))
        return self

    def get_fidx2tok(self, fidx):
        """
        Return the token by the corresponding feature index.
        """

        bases = self._extra_nr_feats
        if len(bases) <= 0 or fidx <= bases[0]:
            idx2tok = self.text_prep.get_idx2tok
            fidx2ngram = self.feat_gen.get_fidx2ngram
            return [idx2tok(idx) for idx in fidx2ngram(fidx)]
        else :
            for i in range(len(self._extra_file_ids)):
                if fidx <= bases[i+1]:
                    return ['{0}:{1}'.format(self._extra_file_ids[i], fidx - bases[i])]

    def toSVM(self, text, class_name = None, extra_svm_feats = []):
        if len(extra_svm_feats) > 0 and self._readonly and len(self._extra_file_ids) != 0 and len(self._extra_file_ids) != len(extra_svm_feats):
            raise ValueError("wrong size of extra_svm_feats")


        text = self.text_prep.preprocess(text)
        feat = self.feat_gen.toSVM(text)
        bases = self._extra_nr_feats
        for i, extra_feat in enumerate(extra_svm_feats):
            for fid in extra_feat:
                if bases[i] + fid > bases[i+1]:
                    continue
                feat[bases[i]+fid] = extra_feat[fid]

        if class_name is None:
            return feat

        return feat, self.getClassIdx(class_name)

    def getClassIdx(self, class_name):
        """
        Return the class index by the class name.
        """
        return self.class_map.toIdx(class_name)

    def getClassName(self, class_idx):
        """
        Return the class name by the class index.
        """
        return self.class_map.toClassName(class_idx)

    def __str__(self):
        return 'Text2svmConverter: ' + (self._option or 'default')


def convert_text(text_src, converter, output=''):
    """
    Convert a text data to a LIBSVM-format data.
    """

    if output == "": output = text_src+'.svm'

    if isinstance(output, str):
        output = open(output,'w')
    elif not isinstance(output, file):
        raise TypeError('output is a str or a file.')

    if isinstance(text_src, str) or isinstance(text_src, file):
        text_src = open(text_src)
        for line in text_src:
            try:
                label, text = line.split('\t', 1)
            # except Exception as e:
            except Exception:
                label, text = '**ILL INST**', '**ILL INST**'
                #raise ValueError('cannot tokenize: ' + line)
            feat, label = converter.toSVM(text, label)
            feat = ''.join(' {0}:{1}'.format(f,feat[f]) for f in sorted(feat))
            if label == None: label = -1
            output.write(str(label) + ' ' +feat+'\n')
        output.close()
        text_src.close()

    elif isinstance(text_src, list):
        for sample in text_src:
            feat, label = converter.toSVM(sample[1], sample[0])
            feat = ''.join(' {0}:{1}'.format(f,feat[f]) for f in sorted(feat))
            if label == None: label = -1
            output.write(str(label) + ' ' +feat+'\n')
        output.close()
