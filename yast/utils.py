#!/usr/bin/env python
# encoding: utf-8

import sys
import os
from ctypes import CDLL, POINTER, c_char_p, c_int, c_int64, c_char

def iterdict(d):
    if sys.version_info[0] >= 3:
        return d.items()
    else:
        return d.iteritems()


def dict2list(d):
    if len(d) == 0:
        return []
    m = max(v for k,v in iterdict(d))
    ret = [''] * (m+1)
    for k,v in iterdict(d):
        ret[v] = k
    return ret


def list2dict(l):
    return dict((v,k) for k,v in enumerate(l))

# XXX This function must support outputing to one of the input file!!
def merge_files(svm_files, offsets, is_training, output):
	if not isinstance(offsets, list) or len(svm_files) != len(offsets):
		raise ValueError('offsets should be a list where the length is the number of merged files')

	util = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'classifier', 'learner', 'util.so.1'))
	util.merge_problems.restype = None
	util.merge_problems.argtypes = [POINTER(c_char_p), c_int, POINTER(c_int64), c_char_p, c_char, POINTER(c_int64)]

	size = len(svm_files)

	c_svm_files = (c_char_p * size)()
	for i, f in enumerate(svm_files):
		c_svm_files[i] = c_char_p(f.encode())
	c_offsets = (c_int64 * size)()
	if not is_training:
		for i, v in enumerate(offsets):
			c_offsets[i] = c_int64(v)
	c_is_training = c_char(chr(is_training).encode('ascii'))
	c_error_code = c_int64()

	output = c_char_p(bytes(output,'utf-8')) if sys.version_info[0] >= 3 else c_char_p(output)
	util.merge_problems(c_svm_files, c_int(size), c_offsets, output, c_is_training, c_error_code)

	error_code = c_error_code.value

	if error_code > 0:
		raise ValueError('wrong file format in line ' + str(error_code))
	elif error_code == -1:
		raise IOError('cannot open file')
	elif error_code == -2:
		raise MemoryError("Memory Exhausted. Try to restart python.")
	elif error_code == -3:
		raise ValueError('merging svm files of different sizes')
	elif error_code == -4:
		raise ValueError('at least one file should be given to merge')

	if is_training:
		for i in range(size):
			offsets[i] = c_offsets[i]




