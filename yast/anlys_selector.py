#!/usr/bin/env python

from random import sample

__all__ = ['wrong', 'with_labels', 'sort_by_dec', 'subset', 'selectorize', 'reverse']

def selectorize(option = 'general', comment = None):
	def inner_func(input_func):
		if option == "select":
			def inner_func2(insts):
				return list(filter(input_func, insts))
		elif option == "sort":
			def inner_func2(insts):
				return sorted(insts, key = input_func)
		elif option == "general":
			inner_func2 = input_func
		else:
			raise Exception("No such setting.")

		if input_func is None or comment is None:
			inner_func2._libshorttext_msg = "user-defined selector function"
		else:
			inner_func2._libshorttext_msg = comment

		inner_func2.__doc__ = input_func.__doc__

		return inner_func2
	return inner_func

@selectorize('select', 'Select wrongly predicted instances')
def wrong(inst):
	return inst.true_y !=  inst.predicted_y

def with_labels(labels, target = 'both'):
	@selectorize('select', 'labels: "{0}"'.format('", "'.join(labels)))
	def inner_func(inst):
		if target == 'both':
			return inst.true_y in labels and inst.predicted_y in labels
		elif target == 'or':
			return inst.true_y in labels or inst.predicted_y in labels
		elif target == 'true':
			return inst.true_y in labels
		elif target == 'predict':
			return inst.predicted_y in labels
		else:
			raise Exception("No such setting.")
	return inner_func

@selectorize('sort', 'Sort by maximum decision values.')
def sort_by_dec(inst):
	return max(inst.decvals)

def subset(amount, method = 'top'):
	@selectorize(comment = 'Select {0} instances in {1}.'.format(amount, method))
	def inner_func(insts):
		if amount > len(insts):
			return insts
		elif method == 'random':
			return sample(insts, amount)
		elif method == 'top':
			return insts[0:amount]
		else:
			raise Exception("No such setting.")
	return inner_func


@selectorize(comment = 'Reverse the order of instances')
def reverse(insts):
	"""
	Reverse the order of instances.

	This function should be passed to :meth:`InstanceSet.select` without any
	argument.
	"""
	return list(reversed(insts))
