import os
import glob
import json
import shutil
import xmltodict

xml_list = glob.glob('surgical_dataset/HarmonicACE_HMARUCDV/*.xml')
os.makedirs('surgical_dataset/HarmonicACE_HMARUCDV/json', exist_ok=True)

for i in xml_list:
    with open(i,'r') as f:
        xmlString = f.read()
 
    print("xml input (xml_to_json.xml):")
    
    jsonString = json.dumps(xmltodict.parse(xmlString), indent=4)
    
    print("\nJSON output(output.json):")
    print(jsonString)
    
    with open("{}.json".format(i), 'w') as f:
        f.write(jsonString)

    fn_org = '{}.json'.format(i.split('/')[2:][0])
    fn_new = '{}.json'.format(i.split('/')[2:][0].split('.')[:-1][0])
    print(fn_new)
    src = 'surgical_dataset/HarmonicACE_HMARUCDV/'
    dir = 'surgical_dataset/HarmonicACE_HMARUCDV/json/'
    shutil.move(src + fn_org, dir + fn_org)
    os.rename(dir + fn_org, dir + '{}'.format(fn_new))