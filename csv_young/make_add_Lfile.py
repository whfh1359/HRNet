import cv2
import pandas as pd

new = pd.read_csv('/app/csv/c_kneebone_fix_total_add_prob_new_make.csv')
target = new[new['letter_L_prob']<0.1]
file_name = list(map(str,target['file_name']))

for i in file_name:
#    file ='00000000' + str(i)
    img = cv2.imread('/app/Classified_Bilateral_v3/'+ i + '.jpg')
    cv2.imwrite('/app/add_L_file/' + i + '.jpg', img)
