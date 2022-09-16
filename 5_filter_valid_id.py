import numpy as np
import shutil
import csv
valid_id = []
with open('./animal_valid_1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if row[3] == "1" :
            valid_id.append(row)
            line_count+=1
    print(f'Processed {line_count} lines.')
filename = './animal_valid_horse.csv'
with open(str(filename), 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in valid_id:
        wr.writerow(i)
print(f'Done')


            
print(valid_id)