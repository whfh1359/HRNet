import cv2
import pandas as pd
import matplotlib as plt

df = pd.read_csv('/app/csv_test/c_kneebone_fix_total_add_prob_2.csv')
df['middle'] = (df['right_medial_x'] + df['left_medial_x'])/2
name = list(df['file_name'])
middle = list(map(int, df['middle']))
right_medial = list(zip(df['right_medial_x'], df['right_medial_y']))
left_medial = list(zip(df['left_medial_x'], df['left_medial_y']))
right_f = list(zip(df['right_f_x'], df['right_f_y']))
left_f = list(zip(df['left_f_x'], df['left_f_y']))
file = list(zip(map(str,name), middle, right_medial, left_medial, right_f, left_f))

#file = list(zip(map(str,name), middle))

r = (255,0,0)
g = (0,255,0)
b = (0,0,255)
w = (255,255,255)

for i in file:
    filename ='00000000' + i[0]
    #img = cv2.imread('/app/data/kneebone/images/val/'+ filename + '.jpg')
    img_reverse = cv2.imread('/app/check/'+ filename + '.jpg')
    #r_leg = img[:, :i[1]].copy()
    #l_leg = img[:, i[1]:].copy()
    #check = cv2.hconcat([l_leg,r_leg])
    #cv2.imwrite('/app/check/' + filename + '.jpg', check)
    #cv2.imwrite('/app/half_test/' + filename + '_r_leg.jpg', r_leg)
    #cv2.imwrite('/app/half_test/' + filename + '_l_leg.jpg', l_leg)
    #p_img = cv2.line(img, i[2], i[2], r, 40)
    #p_img = cv2.line(img, i[3], i[3], g, 40)
    #p_img = cv2.line(img, i[4], i[4], b, 40)
    #p_img = cv2.line(img, i[5], i[5], w, 40)
    rv_p_img = cv2.line(img_reverse, i[2], i[2], r, 40)
    rv_p_img = cv2.line(img_reverse, i[3], i[3], g, 40)
    rv_p_img = cv2.line(img_reverse, i[4], i[4], b, 40)
    rv_p_img = cv2.line(img_reverse, i[5], i[5], w, 40)

    #cv2.imwrite('/app/val_point/' + filename + '.jpg', p_img)
    cv2.imwrite('/app/check_point/' + filename + '.jpg', rv_p_img)


