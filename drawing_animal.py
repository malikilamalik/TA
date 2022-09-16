# a = [[257.,  49.],
#  [255.,  47.],
#  [270.,  90.],
#  [202., 269.],
#  [181., 272.],
#  [164., 262.],
#  [ 80., 262.],
#  [116.,  87.],
#  [204., 221.],
#  [189., 229.],
#  [128., 209.],
#  [ 85., 196.],
#  [204., 108.],
#  [174., 110.],
#  [212., 151.],
#  [169., 158.],
#  [133., 138.],
#  [ 83., 148.]]
# a = [[254, 32], 
#      [242, 35],
#      [261, 63],
#      [201, 74],
#      [79, 80],
#      [221, 116],
#      [258, 112], 
#      [256, 150],
#      [197, 117], 
#      [214, 145], 
#      [232, 177], 
#      [109, 127], 
#      [98, 160],
#      [123, 190],
#      [78, 128],
#      [52, 152],
#      [35, 186]]
# a = [[294., 170.],
#  [297., 170.],
#  [300., 227.],
#  [229., 438.],
#  [189., 441.],
#  [100., 421.],
#  [ 49., 426.],
#  [ 92., 187.],
#  [220., 378.],
#  [195., 378.],
#  [ 86., 338.],
#  [ 61., 329.],
#  [232., 227.],
#  [192., 232.],
#  [243., 289.],
#  [180., 289.],
#  [115., 249.],
#  [ 49., 261.]]
# a = [
#     [306, 170],
#     [273, 165],
#     [295, 215],
#     [221, 232],
#     [68, 195],
#     [235, 312],
#     [221, 371],
#     [224, 427],
#     [184, 308],
#     [184, 369], 
#     [183, 426], 
#     [100, 299], 
#     [91, 343], 
#     [95, 406], 
#     [54, 295], 
#     [49, 345],
#     [45, 413]]
# a = [[971., 373.],
#  [927., 371.],
#  [947., 463.],
#  [781., 517.],
#  [918., 476.],
#  [781., 515.],
#  [783., 517.],
#  [835., 297.],
#  [776., 510.],
#  [749., 502.],
#  [749., 493.],
#  [781., 510.],
#  [805., 322.],
#  [725., 317.],
#  [815., 412.],
#  [722., 415.],
#  [859., 336.],
#  [817., 410.]]
# a = [[411., 327.],
#  [416., 332.],
#  [420., 336.],
#  [425., 323.],
#  [474., 553.],
#  [474., 553.],
#  [474., 553.],
#  [524., 372.],
#  [425., 336.],
#  [434., 336.],
#  [447., 481.],
#  [452., 476.],
#  [416., 336.],
#  [425., 336.],
#  [429., 345.],
#  [425., 345.],
#  [416., 336.],
#  [416.,332.]]
# HORSE
# a = [
#  [294., 170.],
#  [297., 170.],
#  [300., 227.],
#  [229., 438.],
#  [189., 441.],
#  [100., 421.],
#  [ 49., 426.],
#  [ 92., 187.],
#  [220., 378.],
#  [195., 378.],
#  [ 86., 338.],
#  [ 61., 329.],
#  [232., 227.],
#  [192., 232.],
#  [243., 289.],
#  [180., 289.],
#  [115., 249.],
#  [ 49., 261.]]
# a = [[ 38.,  35.],
#  [ 46.,  40.],
#  [ 23.,  69.],
#  [122., 200.],
#  [119., 192.],
#  [235., 197.],
#  [237., 194.],
#  [219.,  37.],
#  [122., 152.],
#  [122., 150.],
#  [245., 142.],
#  [248., 129.],
#  [114.,  63.],
#  [122.,  66.],
#  [117., 100.],
#  [125.,  97.],
#  [240.,  84.],
#  [227.,  84.]]
#Tiger
# a = [[345, 172], [327, 177], [346, 195], [291, 190], [130, 126], [302, 219], [313, 253], [326, 277], [235, 211], [232, 241], [235, 271], [164, 194], [152, 223], [159, 256], [136, 175], [116, 210], [105, 253]]

animal_category = ["horse","argali_sheep","buffalo","cheetah","gorilla","antelope","squirrel","rabbit","monkey"]
from PIL import Image, ImageDraw
import csv
import os
import numpy as np
def line(image_path, output_path, a):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    colors = ["red", "green", "blue", "yellow",
              "purple", "orange", "tomato", "turquoise", "violet",
              "rosybrown", "plum", "pink", "azure", "forestgreen", "limegreen",
              "ivory", "hotpink", "gold"
              ]
    # #1
    # draw.text((a[0][0], a[0][1]) ,"1","red")
    # #2
    # draw.text((a[1][0], a[1][1]) ,"2","green")
    # #3
    # draw.text((a[2][0], a[2][1]) ,"3","blue")
    # #4
    # draw.text((a[3][0], a[3][1]) ,"4","yellow")
    # #5
    # draw.text((a[4][0], a[4][1]) ,"5","purple")
    # #6
    # draw.text((a[5][0], a[5][1]) ,"6","orange")
    # #7
    # draw.text((a[6][0], a[6][1]) ,"7","tomato")
    # #8
    # draw.text((a[7][0], a[7][1]) ,"8","turquoise")
    # #9
    # draw.text((a[8][0], a[8][1]) ,"9","violet")
    # #10
    # draw.text((a[9][0], a[9][1]) ,"10","rosybrown")
    # #11
    # draw.text((a[10][0], a[10][1]) ,"11","plum")
    # #12
    # draw.text((a[11][0], a[11][1]) ,"12","pink")
    # #13
    # draw.text((a[12][0], a[12][1]) ,"13","azure")
    # #14
    # draw.text((a[13][0], a[13][1]) ,"14","forestgreen")
    # #15
    # draw.text((a[14][0], a[14][1]) ,"15","limegreen")
    # #16
    # draw.text((a[15][0], a[15][1]) ,"16","ivory")
    # #17
    # draw.text((a[16][0], a[16][1]) ,"17","hotpink")
    #18
    # draw.text((a[17][0], a[17][1]) ,"18","gold")

    # left Eye To right Eye
    draw.line([(a[0][0], a[0][1]),(a[1][0], a[1][1])], width=2, fill="red")
    # left Eye To Mouth
    draw.line([(a[0][0], a[0][1]),(a[2][0], a[2][1])], width=2, fill="green")
    # right Eye To Mouth
    draw.line([(a[1][0], a[1][1]),(a[2][0], a[2][1])], width=2, fill="blue")

    # front left feet
    draw.line([(a[3][0], a[3][1]),(a[8][0], a[8][1]),(a[14][0], a[14][1])], width=2, fill="yellow")

    # front right feet
    draw.line([(a[4][0], a[4][1]),(a[9][0], a[9][1]),(a[15][0], a[15][1])], width=2, fill="purple")
    image.save(output_path)

    # back left feet
    draw.line([(a[5][0], a[5][1]),(a[10][0], a[10][1]),(a[16][0], a[16][1])], width=2, fill="gold")

    # back right feet
    draw.line([(a[6][0], a[6][1]),(a[11][0], a[11][1]),(a[17][0], a[17][1])], width=2, fill="hotpink")
    image.save(output_path)
if __name__ == "__main__":
    # line("data/000000028824.jpg", "lines9.jpg")
    # line("./valid_data/tiger/000000040994.jpg", "lines8.jpg")
    for i in animal_category:
        # prediction_path = './prediction/animal_{}_pck_score.csv'.format(i)
        # prediction_list = []
        # with open(prediction_path) as csv_file:
        #     csv_reader = csv.reader(csv_file, delimiter=',')
        #     line_count = 0
        #     for row in csv_reader:
        #         a = row
        #         print(list(a[0]))
        #         # prediction_list.append([row[0]])
        prediction_path = 'prediction/animal_{}_pck_score.npy'.format(i)
        prediction_list = np.load(prediction_path)


        valid_path = './valid_annotations/val/valid/animal_{}_id.csv'.format(i)
        valid_list = []
        print(valid_path)
        with open(valid_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                valid_list.append(row)
        k = 0
        for val in valid_list:
            image_path = './valid_data/{}/{}'.format(i,val[3])
            dir_path = './drawing_data/{}'.format(i)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            output_path = dir_path + '/{}'.format(val[3])
            line(image_path, output_path, prediction_list[k])
            k+=1