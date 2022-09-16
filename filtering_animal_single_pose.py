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

landmark_path = os.path.join('valid_annotations')

for animal in animal_category:
    filename = './valid_annotations/val/id/animal_{}_test.csv'.format(animal[1])
    
    all_list = []
    unseen_image_list = []
    unseen_id_list = []
    seen_image_list = []
    valid_list = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            all_list.append(row)

    for i in all_list:
        if i[2] not in unseen_id_list: 
            unseen_id_list.append(i[2])
            unseen_image_list.append(i)
        else:
            seen_image_list.append(i[2])
    
    for k in unseen_image_list:
        if k[2] not in seen_image_list:
            valid_list.append([k[0],k[1],k[2], k[3], 1])

    filename_valid = './valid_annotations/val/valid/animal_{}_id.csv'.format(animal[1])
    
    with open(str(filename_valid), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in valid_list:
            wr.writerow(i)
