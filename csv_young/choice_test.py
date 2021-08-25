import pandas as pd


choice_answer = pd.read_csv('/app/csv/addL/choice_answer.csv')
choice_answer['filename'] = choice_answer['filename']
choice_answer_real_letter_R = list(choice_answer['real_letter_R'])
choice_answer_real_letter_L = list(choice_answer['real_letter_L'])

choice =pd.read_csv('/app/csv/addL/choice.csv')
choice['filename'] = choice['filename']
choice_pred_letter_R = list(choice['pred_letter_R'])
choice_pred_letter_L = list(choice['pred_letter_L'])

total = pd.concat([choice_answer, choice], axis = 1)
wrong = total[(total['real_letter_R'] != total['pred_letter_R']) | (total['real_letter_L'] != total['pred_letter_L'])]
print(wrong)
