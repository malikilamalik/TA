import numpy as np
import shutil
import csv

a = np.load('test.npy')
line_count = 0
filename = './animal_valid_id.csv'
with open(str(filename), 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in a:
        result_1 = str(i[2]).zfill(12)
        img_path = "{}.jpg".format(result_1)
        wr.writerow([i[0],i[1],img_path])
print(f'Done')


            