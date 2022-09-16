import json
import numpy as np
import os

landmark_path = os.path.join('annotations')
frame_num = 0
test_path = ["ap10k-train-split1.json","ap10k-test-split1.json","ap10k-val-split1.json"]
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
    for i,n in enumerate(data['annotations']):
        if n['category_id'] == 21:
            # print(n)
            # result_1 = str(n['image_id']).zfill(12)
            # print("data/{}.jpg".format(result_1))
            list_data.append([k,n['id'],n['image_id']])
            k+=1
    # Closing file
    f.close()

with open('test.npy', 'wb') as f:
    np.save(f, np.array(list_data))