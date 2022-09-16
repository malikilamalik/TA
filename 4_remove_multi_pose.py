import numpy as np
import shutil
import csv
all_image_list = []
duplicated_image_list = []
valid_list = []
all_list = []
with open('./valid_annotations/id/animal_panda_id.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        all_list.append(row)

for i in all_list:
    if i[2] not in str(all_image_list): 
        all_image_list.append(i) 
    else:
        duplicated_image_list.append(i[2])

for i in all_image_list:
    if i[2] not in str(duplicated_image_list):
        valid_list.append(i)
filename = './animal_valid_1.csv'
print(len(valid_list))
with open(str(filename), 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in valid_list:
        wr.writerow([i[0],i[1],i[2],1])
print(f'Done')