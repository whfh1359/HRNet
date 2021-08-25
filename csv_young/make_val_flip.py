import cv2
import pandas as pd
import matplotlib as plt

df = pd.read_csv('/app/csv/c_kneebone_fix_total_add_prob.csv')
name = list(df['file_name'])
file = list(map(str,name))

for i in file:
    filename ='00000000' + i
    img = cv2.imread('/app/data/kneebone/images/val/'+ filename + '.jpg')
    reverse_img = cv2.flip(img, 1)
    cv2.imwrite('/app/check/' + filename + '.jpg', reverse_img)


