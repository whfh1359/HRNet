import pandas as pd
import csv

df_pred = pd.read_csv('/app/csv/predict_addL.csv')
df_pred = df_pred.drop(['Unnamed: 0'], axis = 1)
df_pred = df_pred.sort_values(by = 'file_name', ascending = True)

name_pred = list(df_pred['file_name'])
df_pred['middle'] = (df_pred['right_medial_x'] + df_pred['left_medial_x'])/2
middle_pred = list(map(int, df_pred['middle']))
r_prob_pred = list(df_pred['letter_R_prob'])
l_prob_pred = list(df_pred['letter_L_prob'])
right_medial_pred= list(zip(df_pred['right_medial_x'], df_pred['right_medial_y']))
left_medial_pred = list(zip(df_pred['left_medial_x'], df_pred['left_medial_y']))
right_f = list(zip(df_pred['right_f_x'], df_pred['right_f_y']))
left_f = list(zip(df_pred['left_f_x'], df_pred['left_f_y']))
right_letter = list(zip(df_pred['letter_R_x'], df_pred['letter_R_y']))
left_letter = list(zip(df_pred['letter_L_x'], df_pred['letter_L_y']))
letter_R_x_pred = list(df_pred['letter_R_x'])
letter_L_x_pred = list(df_pred['letter_L_x'])

file_view_pred = list(zip(name_pred, middle_pred, right_medial_pred, left_medial_pred, right_f, left_f, r_prob_pred, l_prob_pred, right_letter, left_letter))

answer = []
choice = [] # 0,1 이 오,왼, 만약 1,0 이면 왼오
choice_answer = []
answer.append(['filename','pred_letter_R', 'pred_letter_L'])
choice.append(['filename','pred_letter_R', 'pred_letter_L'])
choice_answer.append(['filename','real_letter_R', 'real_letter_L'])

for i in range(len(name_pred)):
#for i in range(10):
#    print(i)
    temp = max(r_prob_pred[i], l_prob_pred[i])
    if (temp < 0.55):
        answer.append([name_pred[i],0,0])
    elif ((r_prob_pred[i]<0.55)&(l_prob_pred[i] > 0.8) & (middle_pred[i] < letter_L_x_pred[i])):
        answer.append([name_pred[i], 0, 1])
    elif ((r_prob_pred[i] > 0.55)&(l_prob_pred[i] < 0.8)&(middle_pred[i] > letter_R_x_pred[i])):
        answer.append([name_pred[i], 1, 0])
    else:
        answer.append([name_pred[i], 1, 1])

for i in range(len(name_pred)):
    temp = max(r_prob_pred[i], l_prob_pred[i])
    if temp > 0.46:
        choice_answer.append([name_pred[i], 0, 1])
    if temp < 0.46:
        choice_answer.append([name_pred[i], 1000, 1000])

    if temp < 0.46:
        choice.append([name_pred[i], 1000, 1000])

    elif r_prob_pred[i] > l_prob_pred [i]:
        if letter_R_x_pred[i] < middle_pred[i]:
            choice.append([name_pred[i], 0, 1])
        else:
            choice.append([name_pred[i], 1, 0])

    else:
        if letter_L_x_pred[i] > middle_pred[i]:
            choice.append([name_pred[i], 0, 1])
        else:
            choice.append([name_pred[i], 1, 0])


#answer
with open('/app/csv/addL/predict_val_best.csv', 'w') as file:
    write = csv.writer(file)
    write.writerows(answer)

with open('/app/csv/addL/choice_answer.csv', 'w') as file:
    write = csv.writer(file)
    write.writerows(choice_answer)

with open('/app/csv/addL/choice.csv', 'w') as file:
    write = csv.writer(file)
    write.writerows(choice)