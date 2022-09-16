import numpy as np
import shutil
import csv
import os
#Looping through Categories
animal_category = []
with open('./animal_categories.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        animal_category.append(row)

for animal in animal_category:
    filename = './valid_annotations/val/valid/animal_{}_test.csv'.format(animal[1])
    data_file = []
    line_count = 0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            data_file.append(row)
    dst = './valid_data/{}'.format(animal[1])
    if not os.path.exists(dst):
        os.mkdir(dst)
    for i in data_file:
        line_count+=1
        result_1 = str(i[2]).zfill(12)

        src = './data/{}.jpg'.format(result_1)
        shutil.copy(src, dst)
    print(f'Processed {line_count} lines.')