import os, sys
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree, dump
import json

path = '/app/separate_img_point/'
point_list = ['right_medial','right_f','left_medial','left_f','letter_R','letter_L'] # fm tl tm
file_list = [file for file in os.listdir(path) if file.endswith(".json")]
for filename in file_list:
    print(filename)
    with open(os.path.join(path, filename)) as f:
        json_object = json.load(f)
    # json_object = json.loads(os.path.join(path, filename))
    # print(json_object['imagePath'])
    # print('fl')
    # print(json_object['shapes'][0]['points'][0])
    json_result = {}
    labelname,xmin,ymin = [],[],[]
    for _ in json_object['shapes']:
        if _['label'] == "left":
            _['label'] = "letter_L"
        if _['label'] == "right":
            _['label'] = "letter_R"
        labelname.append('<name>' + _['label'] + '</name>')
        xmin.append('<xmin>' + str(_['points'][0][0]) + '</xmin>')
        # print(xmin)
        ymin.append('<ymin>' + str(_['points'][0][1]) + '</ymin>')
    with open('/app/knee_key_point/json_new/train' + '/' +
              filename.split('.json')[0] + '.xml', "w") as file:
        #        json_object['imagePath'].split('.jpg')[0] + '.xml', "w") as file:
        for i in range(len(labelname)):
            file.write(labelname[i] + "\n")
            file.write(xmin[i] + "\n")
            file.write(ymin[i] + "\n")

# assert json_object['id'] == 1
# assert json_object['email'] == 'Sincere@april.biz'
# assert json_object['address']['zipcode'] == '92998-3874'
# assert json_object['admin'] is False
# assert json_object['hobbies'] is None