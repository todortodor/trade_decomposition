#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 15:11:16 2024

@author: simonl
"""

import csv
 
 
def write_csv(filename, data, header=None):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        if header:
            writer.writerow(header)
        writer.writerows(data)
 
 
def split_csv(filename, num_rows, has_header=True):
    name, extension = filename.split('.')
    file_no = 1
    chunk = []
    row_count = 0
    header = ''
 
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if has_header:
                header = row
                has_header = False
                continue
            chunk.append(row)
            row_count += 1
            if row_count >= num_rows:
                write_csv(f'{name}-{file_no}.{extension}', chunk, header)
                chunk = []
                file_no += 1
                row_count = 0
        if chunk:
            write_csv(f'{name}-{file_no}.{extension}', chunk, header)
            
split_csv('data/yearly_CSV_agg_treated/datas2018/input_output_2018.csv', 300000)