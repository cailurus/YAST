#!/usr/bin/env python
# encoding: utf-8

def write(string, output = None):
	if output is None:
		print(string)
	else:
		output.write(string + '\n')

def draw_table(rows, columns, table, output = None):
	offset = 2
	column_widths = []
	title_width = max([len(row) for row in rows]) + offset

	for col_idx, column in enumerate(columns):
		column_widths.append(max([len(table[row_idx][col_idx]) \
				for row_idx, row in enumerate(rows)] + [len(column)]) + offset)

	string = ''.ljust(title_width)
	for idx, column in enumerate(columns):
		string += column.rjust(column_widths[idx])
	write(string, output)

	for row_idx, row in enumerate(rows):
		string = row.ljust(title_width)
		for col_idx, column in enumerate(columns):
			string += table[row_idx][col_idx].rjust(column_widths[col_idx])
		write(string, output)
