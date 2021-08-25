import cv2
import pandas as pd
import csv
import math

df_pred = pd.read_csv('/app/csv/predict_addL.csv')
df_pred = df_pred.drop(['Unnamed: 0'], axis = 1)
df_pred = df_pred.sort_values(by = 'file_name', ascending = True)

name = list(df_pred['file_name'])
r_prob = list(df_pred['letter_R_prob'])
l_prob = list(df_pred['letter_L_prob'])
df_pred['middle'] = (df_pred['right_medial_x'] + df_pred['left_medial_x'])/2
middle = list(map(int, df_pred['middle']))
right_medial = list(zip(df_pred['right_medial_x'], df_pred['right_medial_y']))
left_medial = list(zip(df_pred['left_medial_x'], df_pred['left_medial_y']))
right_f = list(zip(df_pred['right_f_x'], df_pred['right_f_y']))
left_f = list(zip(df_pred['left_f_x'], df_pred['left_f_y']))
right_letter = list(zip(df_pred['letter_R_x'], df_pred['letter_R_y']))
left_letter = list(zip(df_pred['letter_L_x'], df_pred['letter_L_y']))
file_view_pred = list(zip(name, middle, right_medial, left_medial, right_f, left_f, r_prob, l_prob, right_letter, left_letter))


answer_val = pd.read_csv('/app/csv/answer_val_addL.csv')
answer_val['filename'] = answer_val['filename'].add(1942)
answer_val_real_letter_R = list(answer_val['real_letter_R'])
answer_val_real_letter_L = list(answer_val['real_letter_L'])

predict_val =pd.read_csv('/app/csv/addL/predict_val_best.csv')
predict_val.rename(columns = {'filename' : 'pred_filename'}, inplace = True)
predict_val_pred_letter_R = list(predict_val['pred_letter_R'])
predict_val_pred_letter_L = list(predict_val['pred_letter_L'])


total = pd.concat([answer_val, predict_val], axis = 1)
wrong = total[(total['real_letter_R'] != total['pred_letter_R']) | (total['real_letter_L'] != total['pred_letter_L'])]
#print(wrong)

####성능 테스트####
accuracy = total[(total['real_letter_R'] == total['pred_letter_R']) & (total['real_letter_L'] == total['pred_letter_L'])]
accuracy = list(accuracy['pred_filename'])


print("accuracy = ", len(accuracy)/831)

name_list = list(wrong['pred_filename'])
file_view_pred = list(file_view_pred)
print(len(name_list))

temp_2 = []

for i in range(len(file_view_pred)):
    if int(file_view_pred[i][0]) in name_list:
        temp_2.append(file_view_pred[i])

#print(temp_2)

answer_distance = pd.read_csv('/app/csv/answer_distance_addL.csv')
answer_distance['filename'] = answer_distance['filename'].add(1942)

right_x_dist_answer= list(answer_distance['right_medial_x'])
right_y_dist_answer= list(answer_distance['right_medial_y'])
right_f_x_dist_answer= list(answer_distance['right_f_x'])
right_f_y_dist_answer= list(answer_distance['right_f_y'])

benchmark = []
for i in range(len(right_x_dist_answer)):
    result = math.sqrt(math.pow(right_x_dist_answer[i] - right_f_x_dist_answer[i], 2) + math.pow(right_y_dist_answer[i] - right_f_y_dist_answer[i], 2))
    benchmark.append(int(result*0.25)) # 거리는 1/4

letter_R_x_dist_answer= list(answer_distance['letter_R_x'])
letter_R_y_dist_answer= list(answer_distance['letter_R_y'])
letter_L_x_dist_answer= list(answer_distance['letter_L_x'])
letter_L_y_dist_answer= list(answer_distance['letter_L_y'])
letter_R_x_dist_pred= list(df_pred['letter_R_x'])
letter_R_y_dist_pred= list(df_pred['letter_R_y'])
letter_L_x_dist_pred= list(df_pred['letter_L_x'])
letter_L_y_dist_pred= list(df_pred['letter_L_y'])

dist_R= []
for i in range(len(letter_R_x_dist_answer)):
    result = math.sqrt(math.pow(letter_R_x_dist_answer[i] - letter_R_x_dist_pred[i], 2) + math.pow(letter_R_y_dist_answer[i] - letter_R_y_dist_pred[i], 2))
    dist_R.append(int(result))

dist_L= []
for i in range(len(letter_L_x_dist_answer)):
    result = math.sqrt(math.pow(letter_L_x_dist_answer[i] - letter_L_x_dist_pred[i], 2) + math.pow(letter_L_y_dist_answer[i] - letter_L_y_dist_pred[i], 2))
    dist_L.append(int(result))


r = (255,0,0)
g = (0,255,0)
b = (0,0,255)
w = (255,255,255)

font = cv2.FONT_HERSHEY_DUPLEX
cnt = 0

for i in temp_2:
    filename = '00000000' + str(i[0])
    print(filename)
    print('file:', int(filename)-1942)
    img = cv2.imread('/app/data/kneebone/images/val/' + filename + '.jpg')
    temp = max(i[6], i[7])
    print("r_prob, l_prob: ", i[6], i[7])
    print("right_letter, left_letter: ", i[8], i[9])
    draw_r = (int(i[1] / 2), 1700)
    draw_l = (int(draw_r[0]) + i[1], 1700)
    index = i[0] - 1943
#    print(index)
    print("answer_val_R:", answer_val_real_letter_R[index], "answer_val_L:",  answer_val_real_letter_L[index] )
    print("predict_val_R:", predict_val_pred_letter_R[index], "predict_val_L:",  predict_val_pred_letter_L[index] )

    print(draw_r, draw_l)
    if (temp < 0.5): # 둘다 0.5보다 낮은 확률들은 다 모르는 것(L,R 구분)
        print("I don't know")
        cv2.putText(img, "I don't know", draw_r, font, 2, (0, 0, 155), 2, cv2.LINE_AA)
        cv2.putText(img, "I don't know", draw_l, font, 2, (0, 0, 155), 2, cv2.LINE_AA)
        cv2.putText(img, "NOT R", i[8], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
        cv2.putText(img, "NOT L", i[9], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
    elif (i[6] < i[7]):
        print("R draw")
        cv2.putText(img, "R", draw_r, font, 2, (0, 0, 155), 2, cv2.LINE_AA)
        if (dist_L[index] <= benchmark[index]):
            cv2.putText(img, "L=Good", i[9], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
            cnt +=1

        else:
            cv2.putText(img, "L=Bad...", i[9], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
    elif (i[6] > i[7]):
        print("L draw")
        cv2.putText(img, "L", draw_l, font, 2, (0, 0, 155), 2, cv2.LINE_AA)
        if (dist_R[index] <= benchmark[index]):
            cv2.putText(img, "R=Good", i[8], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
            cnt += 1
        else:
            cv2.putText(img, "R=Bad...", i[8], font, 2, (0, 0, 155), 2, cv2.LINE_AA)

    p_img = cv2.line(img, i[2], i[2], r, 40)
    p_img = cv2.line(img, i[3], i[3], g, 40)
    p_img = cv2.line(img, i[4], i[4], b, 40)
    p_img = cv2.line(img, i[5], i[5], w, 40)
    predict_val_pred_letter_R[index], "predict_val_L:", predict_val_pred_letter_L[index]
    if ((i[6] < 0.5) & (answer_val_real_letter_R[index] != predict_val_pred_letter_R[index])):
        cv2.putText(img, "N", i[8], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
    if ((i[6] >= 0.5) & (answer_val_real_letter_R[index] != predict_val_pred_letter_R[index])):
        cv2.putText(img, "S", i[8], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
    if ((i[7] < 0.5) & (answer_val_real_letter_L[index] != predict_val_pred_letter_L[index])):
        cv2.putText(img, "N", i[9], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
    if ((i[7] >= 0.5) & (answer_val_real_letter_L[index] != predict_val_pred_letter_L[index])):
        cv2.putText(img, "S", i[9], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
    if ((i[6] >= 0.5) & (answer_val_real_letter_R[index] == predict_val_pred_letter_R[index])):
        cv2.putText(img, "R", i[8], font, 2, (0, 0, 155), 2, cv2.LINE_AA)
    if ((i[7] >= 0.5) & (answer_val_real_letter_L[index] == predict_val_pred_letter_L[index])):
        cv2.putText(img, "L", i[9], font, 2, (0, 0, 155), 2, cv2.LINE_AA)

    print("cnt: ", cnt)
    #p_img = cv2.line(img, i[8], i[8], r, 40)
    #p_img = cv2.line(img, i[9], i[9], b, 40)

    cv2.imwrite('/app/csv/test/' + filename + '.jpg', p_img)


