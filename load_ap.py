import json
import numpy as np
import os

# # Opening JSON file
# f = open('ap10k-val-split3.json')
  
# # returns JSON object as 
# # a dictionary
# data = json.load(f)
  
# # Iterating through the json
# # list
# for i,n in enumerate(data['categories']):
#     print(i,n)
  
# # Closing file
# f.close()

landmark_path = os.path.join('annotations')
frame_num = 0
train_path = ["ap10k-train-split1.json","ap10k-test-split1.json","ap10k-val-split3.json"]
test_path = ["ap10k-val-split1.json","ap10k-val-split2.json","ap10k-val-split3.json"]
# animal_category = 0
# if self.animal[0] == 'antelope':
#     animal_category = 1
# elif self.animal[0] == 'argali':
#     animal_category = 2
# elif self.animal[0] == 'bison':
#     animal_category = 3

frame_num = 0
list = []      
for data_path in train_path:
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
        frame_num+=1
        # if n['category_id'] == 21:
        #     print(n)
        #     result_1 = str(n['image_id']).zfill(12)
        #     print("data/{}.jpg".format(result_1))
        #     k = 0
        #     coord = []
        #     vis = []
        #     k_coord = []
        #     for i,l in enumerate(n['keypoints']):
        #         if k != 2 :
        #             k_coord.append(float(l))
        #             k+=1
        #         elif k == 2:
        #             coord.append(k_coord)   
        #             if l != 0 :
        #                 vis.append([1])
        #             else:
        #                 vis.append([l])
        #             k_coord = []
        #             k=0
        #         if i == 50:
        #             coord.append([float(0),float(0)])
        #             vis.append([0])
            # print("cord",coord)
            # print("vis",vis)
            # landmark = np.hstack((coord, vis))
            # landmark_18 = landmark[:18, :]
            # landmark_18 = landmark_18[np.array([1, 2, 3, 8, 11, 14, 17, 5, 7, 10, 13, 16, 18, 18, 6, 9, 12, 15]) - 1]
            # print("landmark",landmark)
            # print("landmark_18",landmark_18)
    # Closing file
    f.close()
print(frame_num)