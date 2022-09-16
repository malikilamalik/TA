import numpy as np

a = np.load('prediction/animal_horse_pck_score.npy')
count = a

for i in a:
    print(i)