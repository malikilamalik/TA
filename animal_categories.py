import json
import csv
import os

landmark_path = os.path.join('annotations')
frame_num = 0
test_path = ["ap10k-train-split1.json"]
list_data = []
k = 0
for data_path in test_path:
    anno_path = os.path.join(landmark_path, data_path)
    # Opening JSON file
    f = open(anno_path)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # for i,n in enumerate(data['categories']):
    #     print(n)
    # Iterating through the json
    # list
    for n in data['categories']:
        list_data.append([n['id'],n['name'],n['supercategory']])
        k+=1
    # Closing file
    f.close()

filename = './animal_categories.csv'
with open(str(filename), 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in list_data:
        wr.writerow(i)
print(f'Done')