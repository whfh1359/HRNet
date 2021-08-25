import pandas as pd
import csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


answer_val = pd.read_csv('/app/csv/answer_val_addL.csv')
answer_val['filename'] = answer_val['filename'].add(1943)
answer_val_real_letter_R = list(answer_val['real_letter_R'])
answer_val_real_letter_L = list(answer_val['real_letter_L'])

#predict_val =pd.read_csv('/app/csv/predict_val_0.6.csv')
#threshold = [0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
#threshold = []
#result = [['Threshold','sens_R','sens_L','Accuracy','TP_R','TN_R','FP_R','FN_R','TP_L','TN_L','FP_L','FN_L']]
result = [['sens_R','sens_L','Accuracy','TP_R','TN_R','FP_R','FN_R','TP_L','TN_L','FP_L','FN_L']]

#result_score = [['Threshold','R_accuracy_score','R_recall_score','R_precision_score','R_f1_score',
#                 'L_accuracy_score','L_recall_score','L_precision_score','L_f1_score' ]]

result_score = [['R_accuracy_score','R_recall_score','R_precision_score','R_f1_score',
                 'L_accuracy_score','L_recall_score','L_precision_score','L_f1_score' ]]


for i in range(1):
    #predict_val =pd.read_csv('/app/csv/addL/predict_val_' + str(threshold[i]) +'.csv')
    predict_val = pd.read_csv('/app/csv/addL/predict_val_best.csv')
    predict_val.rename(columns = {'filename' : 'pred_filename'}, inplace = True)
    predict_val_pred_letter_R = list(predict_val['pred_letter_R'])
    predict_val_pred_letter_L = list(predict_val['pred_letter_L'])


    total = pd.concat([answer_val, predict_val], axis = 1)
    wrong = total[(total['real_letter_R'] != total['pred_letter_R']) | (total['real_letter_L'] != total['pred_letter_L'])]


    ####성능 테스트####
    total_accuracy = total[(total['real_letter_R'] == total['pred_letter_R']) & (total['real_letter_L'] == total['pred_letter_L'])]
    total_accuracy = list(total_accuracy['pred_filename'])

    TP_R = total[(total['real_letter_R']==1)& (total['pred_letter_R']==1)]
    TN_R = total[(total['real_letter_R']==0)& (total['pred_letter_R']==0)]
    FP_R = total[(total['real_letter_R']==0)& (total['pred_letter_R']==1)]
    FN_R = total[(total['real_letter_R']==1)& (total['pred_letter_R']==0)]

    TP_L = total[(total['real_letter_L']==1) &(total['pred_letter_L']==1)]
    TN_L = total[(total['real_letter_L']==0) &(total['pred_letter_L']==0)]
    FP_L = total[(total['real_letter_L']==0) &(total['pred_letter_L']==1)]
    FN_L = total[(total['real_letter_L']==1) &(total['pred_letter_L']==0)]

    TP_R = list(TP_R['pred_filename'])
    TN_R = list(TN_R['pred_filename'])
    FP_R = list(FP_R['pred_filename'])
    FN_R = list(FN_R['pred_filename'])

    TP_L = list(TP_L['pred_filename'])
    TN_L = list(TN_L['pred_filename'])
    FP_L = list(FP_L['pred_filename'])
    FN_L = list(FN_L['pred_filename'])

    count_TP_R = len(TP_R)
    print('count_TP_R: ',count_TP_R)
    count_TN_R = len(TN_R)
    print('count_TN_R: ',count_TN_R)
    count_FP_R = len(FP_R)
    print('count_FP_R: ',count_FP_R)
    count_FN_R = len(FN_R)
    print('count_FN_R: ',count_FN_R)

    count_TP_L = len(TP_L)
    print('count_TP_L: ',count_TP_L)
    count_TN_L = len(TN_L)
    print('count_TN_L: ',count_TN_L)
    count_FP_L = len(FP_L)
    print('count_FP_L: ',count_FP_L)
    count_FN_L = len(FN_L)
    print('count_FN_L: ',count_FN_L)

    sens_R = count_TP_R / (count_TP_R + count_FN_R)
    spec_R = count_TN_R / (count_TN_R + count_FP_R)
    sens_L = count_TP_L / (count_TP_L + count_FN_L)
    spec_L = count_TN_L / (count_TN_L + count_FP_L)

    sens_R = round(sens_R,5)
    spec_R = round(spec_R,5)
    sens_L = round(sens_L,5)
    spec_L = round(spec_L,5)

    Acur = (len(total_accuracy)/831)*100
    Acur = round(Acur, 5)

#    labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]  # 실제 labels
#    guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]  # 에측된 결과

    answer_val_real_letter_R = list(answer_val['real_letter_R'])
    answer_val_real_letter_L = list(answer_val['real_letter_L'])
    predict_val_pred_letter_R = list(predict_val['pred_letter_R'])
    predict_val_pred_letter_L = list(predict_val['pred_letter_L'])


    #print("threshold : ", threshold[i])
    print("total_accuracy= ",(len(total_accuracy)),"/ 831 ========>", (len(total_accuracy)/831)*100,"%")
    print("R의 민감도(Sensitivity)=", sens_R)
    print("R의 특이도(Specificity)=", spec_R)
    print("**********************************")
    print("L의 민감도(Sensitivity)=", sens_L)
    print("L의 특이도(Specificity)=", spec_L)

    R_accuracy_score= round(accuracy_score(answer_val_real_letter_R, predict_val_pred_letter_R),5)
    R_recall_score= round(recall_score(answer_val_real_letter_R, predict_val_pred_letter_R),5)
    R_precision_score= round(precision_score(answer_val_real_letter_R, predict_val_pred_letter_R),5)
    R_f1_score= round(f1_score(answer_val_real_letter_R, predict_val_pred_letter_R),5)
    L_accuracy_score= round(accuracy_score(answer_val_real_letter_L, predict_val_pred_letter_L),5)
    L_recall_score= round(recall_score(answer_val_real_letter_L, predict_val_pred_letter_L),5)
    L_precision_score= round(precision_score(answer_val_real_letter_L, predict_val_pred_letter_L),5)
    L_f1_score= round(f1_score(answer_val_real_letter_L, predict_val_pred_letter_L),5)

    print("R_accuracy_score= ",R_accuracy_score)  # 0.3
    print("R_recall_score= ",R_recall_score)  # 0.42
    print("R_precision_score= ",R_precision_score)  # 0.5
    print("R_f1_score= ",R_f1_score)  # 0.46

    print("L_accuracy_score= ",L_accuracy_score)  # 0.3
    print("L_recall_score= ",L_recall_score)  # 0.42
    print("L_precision_score= ",L_precision_score)  # 0.5
    print("L_f1_score= ",L_f1_score)  # 0.46

    result.append([sens_R, sens_L, Acur, count_TP_R, count_TN_R,
                   count_FP_R, count_FN_R, count_TP_L, count_TN_L, count_FP_L, count_FN_L])
    result_score.append([R_accuracy_score, R_recall_score, R_precision_score, R_f1_score,
                     L_accuracy_score, L_recall_score, L_precision_score, L_f1_score])

'''
    result.append([threshold[i],sens_R,sens_L, Acur,count_TP_R,count_TN_R,
                   count_FP_R,count_FN_R,count_TP_L,count_TN_L,count_FP_L,count_FN_L])
    result_score.append([threshold[i],R_accuracy_score,R_recall_score,R_precision_score,R_f1_score,
                 L_accuracy_score,L_recall_score,L_precision_score,L_f1_score])
'''

with open('/app/csv/addL/result_addL_best.csv', 'w') as file:
    write = csv.writer(file)
    write.writerows(result)

with open('/app/csv/addL/result_score_addL_best.csv', 'w') as file:
    write = csv.writer(file)
    write.writerows(result_score)