import json
import csv
import os

landmark_path = os.path.join('annotations')
frame_num = 0
dataset_path = ["ap10k-train-split1.json","ap10k-test-split1.json","ap10k-val-split1.json"]
list_data = []
#Looping through Dataset
for data_path in dataset_path:
    anno_path = os.path.join(landmark_path, data_path)
    # Opening JSON file
    f = open(anno_path)
    
    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    # Iterating through the json
    # list
    for i,n in enumerate(data['annotations']):
        list_data.append([n['id'],n['image_id'],n['category_id']])
    
    # Closing file
    f.close()

#Looping through Categories
animal_category = []
with open('./animal_categories.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        animal_category.append(row)


for i in animal_category:
    animal_data = []
    k = 0
    for j in list_data:
        if str(i[0]) == str(j[2]):
            animal_data.append(j)
    filename = './valid_annotations/val/id/animal_{}_test.csv'.format(i[1])
    with open(str(filename), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in animal_data:
            result_1 = str(i[1]).zfill(12)
            img_path = "{}.jpg".format(result_1)
            wr.writerow([k, i[0], i[1], img_path])
            k+=1
    print(f'Done')

# print(list_data)