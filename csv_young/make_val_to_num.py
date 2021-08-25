import json, os, math, shutil, cv2, csv
# -*- coding: utf-8 -*-
point_list = ['right_medial','right_f','left_medial','left_f','letter_R','letter_L'] # fm tl tm
img_path = '/home/crescom/deep-high-resolution-net.pytorch2/separate_img_point' + '/'
train_save_path = '/home/crescom/deep-high-resolution-net.pytorch2/data/kneebone/images/train' + '/'
val_save_path = '/home/crescom/deep-high-resolution-net.pytorch2/data/kneebone/images/val' + '/'
xml_path = '/home/crescom/deep-high-resolution-net.pytorch2/knee_key_point/json_new/train' + '/'
save_path = '/home/crescom/deep-high-resolution-net.pytorch2/data/kneebone/annotations' + '/'

xml_list = sorted(os.listdir(xml_path))
answer = []
answer_distance = []
answer.append(['filename','real_letter_R', 'real_letter_L'])
answer_distance.append(['filename','right_medial_x','right_medial_y','Y/N','right_f_x','right_f_y','Y/N','left_medial_x','left_medial_y',
                       'Y/N','left_f_x','left_f_y','Y/N','letter_R_x','letter_R_y','Y/N','letter_L_x','letter_L_y','Y/N'])
def coco_keypoint():
    split_cnt = math.ceil(len(xml_list) * 0.7)
    train_xml_list = xml_list[:split_cnt]
    #train_xml_list = xml_list[split_cnt:]
#
    id_cnt = 0
    ann_id = 1000000


    for one_xml in train_xml_list:
        xml_dict = {}
        with open(xml_path + one_xml,'r') as file:
            lines = file.readlines()
            for one_line in lines:
                if '<name>' in one_line:
                    class_name = one_line.replace('name','').replace('/','').replace('<','').replace('>','').replace(' ','').replace('\t','').replace('\n','')
                    xml_dict[class_name] = []
                elif 'xmin' in one_line:
                    xmin = one_line.replace('xmin', '').replace('/', '').replace('<', '').replace('>', '').replace(' ','').replace('\t','').replace('\n','')
                    xml_dict[class_name].append(int(float(xmin)))
                elif 'ymin' in one_line:
                    ymin = one_line.replace('ymin', '').replace('/', '').replace('<', '').replace('>', '').replace(' ','').replace('\t','').replace('\n','')
                    xml_dict[class_name].append(int(float(ymin)))

        #img_name = one_xml.replace('.xml','.jpg')
        img = cv2.imread(img_path + one_xml)
#        print(img_path+one_xml)
#        img_y, img_x = img.shape[0], img.shape[1]

        id_cnt += 1
        ann_id += 1

#        train_y_dict['images'].append({'file_name':str(id_cnt) + '.jpg', 'id': id_cnt, 'height':img_y, 'width':img_x})
        num_keypoint = 0

#        csv_write_info.append([img_name, str(id_cnt) + '.jpg'])

        temp_xml_list = []
        temp_x = []
        temp_y = []
        temp_xml_list.append(id_cnt)
        for one_point_name in point_list:
            if one_point_name in xml_dict.keys():
#                temp_xml_list.append(id_cnt) #수정한거, 원래는 지워야한다.
                temp_xml_list.append(xml_dict[one_point_name][0])
                temp_xml_list.append(xml_dict[one_point_name][1])
                temp_xml_list.append(2)
                num_keypoint += 1

                temp_x.append(xml_dict[one_point_name][0])
                temp_y.append(xml_dict[one_point_name][1])
            else:
#                temp_xml_list.append(id_cnt) #수정한거, 원래는 지워야한다.
                temp_xml_list.append(0)
                temp_xml_list.append(0)
                temp_xml_list.append(0)
#        print(temp_xml_list)
#        answer_distance.append(id_cnt)
        answer_distance.append(temp_xml_list)

        letter = []
        letter.append(id_cnt)

        for i in range(14,18,3):
#            print(temp_xml_list[i])
#            answer_distance.append(temp_xml_list[i])
            if temp_xml_list[i] == 0:
                letter.append(0)
            else:
                letter.append(1)
        answer.append(letter)
#    print(answer)
#    print(len(answer))
#        print(id_cnt)
    print(answer_distance[1])

    with open('/home/crescom/deep-high-resolution-net.pytorch2/csv/answer_train_1.csv', 'w') as file:
        write = csv.writer(file)
        write.writerows(answer)
    #with open('/home/crescom/deep-high-resolution-net.pytorch2/csv/answer_train_addL.csv', 'w') as file:
    #    write = csv.writer(file)
    #    write.writerows(answer)

    with open('/home/crescom/deep-high-resolution-net.pytorch2/csv/answer_train_2.csv', 'w') as file:
        write = csv.writer(file)
        write.writerows(answer_distance)


coco_keypoint()