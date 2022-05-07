import numpy as np
import pandas as pd
import json
from glob import glob
import os
from tqdm import tqdm
#from preprocess import preprocess
import re
def preprocess(text):
    #text = text.strip()
    #return re.sub('\W+',' ', re.sub('\n',' ',text))
    return text.replace('\r','')

train_data = json.loads(open('./data/train.json').read())
val_data = json.loads(open('./data/val_data.json').read())

def train_prev_labels():
    train_df = pd.DataFrame(columns=['id','start','end','text','prev_1','prev_2','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            #if(len(j['value']['labels'])>1): print(j['value']['labels'])
            if(index==0):
                train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                            i['annotations'][0]['result'][j]['value']['start'],
                                            i['annotations'][0]['result'][j]['value']['end'],
                                            preprocess(i['annotations'][0]['result'][j]['value']['text']),
                                            '[UNK]',
                                            '[UNK]',
                                            i['annotations'][0]['result'][j]['value']['labels'][0]]
            elif(index==1):
                train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                            i['annotations'][0]['result'][j]['value']['start'],
                                            i['annotations'][0]['result'][j]['value']['end'],
                                            preprocess(i['annotations'][0]['result'][j]['value']['text']),
                                            i['annotations'][0]['result'][j-1]['value']['labels'][0],
                                            '[UNK]',
                                            i['annotations'][0]['result'][j]['value']['labels'][0]]
            else:
                train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                            i['annotations'][0]['result'][j]['value']['start'],
                                            i['annotations'][0]['result'][j]['value']['end'],
                                            preprocess(i['annotations'][0]['result'][j]['value']['text']),
                                            i['annotations'][0]['result'][j-1]['value']['labels'][0],
                                            i['annotations'][0]['result'][j-2]['value']['labels'][0],
                                            i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
                
    train_df.to_csv('./data/train_data_prev_label.csv',index=False)

def json_to_csv():

    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('./data/excpetion_in_data_train.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        for j in i['annotations'][0]['result']:
            #id, start, end, text, label
            if(len(j['value']['labels'])>1): print(j['value']['labels'])
            train_df.loc[len(train_df)] = [j['id'],j['value']['start'],j['value']['end'],preprocess(j['value']['text']),j['value']['labels'][0]]

    train_df.to_csv('./data/train_data.csv',index=False)

    val_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in val_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data_val.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        for j in i['annotations'][0]['result']:
            #id, start, end, text, label
            if(len(j['value']['labels'])>1): print(j['value']['labels'])
            val_df.loc[len(val_df)] = [j['id'],j['value']['start'],j['value']['end'],preprocess(j['value']['text']),j['value']['labels'][0]]

    val_df.to_csv('./data/val_data.csv',index=False)




def train_append_labels():

    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = '[UNK]'
                prev_2 = '[UNK]'
            elif(index==1):
                prev_1 = '[UNK]'
                prev_2 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            else:
                prev_1 = i['annotations'][0]['result'][j-2]['value']['labels'][0]
                prev_2 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            s = prev_1 + '[SEP]' + prev_2 + '[SEP]' + preprocess(i['annotations'][0]['result'][j]['value']['text'])
            #if(len(j['value']['labels'])>1): print(j['value']['labels'])
            train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                        i['annotations'][0]['result'][j]['value']['start'],
                                        i['annotations'][0]['result'][j]['value']['end'],
                                        s,
                                        i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    train_df.to_csv('./data/train_data_append_label.csv',index=False)


def train_append_sentence():
    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
            elif(index==1):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])

            s = prev_1 + '[SEP]' + prev_2 + '[SEP]'  + preprocess(i['annotations'][0]['result'][j]['value']['text'])
            #if(len(j['value']['labels'])>1): print(j['value']['labels'])
            train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                        i['annotations'][0]['result'][j]['value']['start'],
                                        i['annotations'][0]['result'][j]['value']['end'],
                                        s,
                                        i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    train_df.to_csv('./data/train_data_append_sentence.csv',index=False)

def val_append_sentence():
    val_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in val_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
            elif(index==1):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])

            s = prev_1 + '[SEP]' + prev_2 + '[SEP]'  + preprocess(i['annotations'][0]['result'][j]['value']['text'])
            #if(len(j['value']['labels'])>1): print(j['value']['labels'])
            val_df.loc[len(val_df)] = [i['annotations'][0]['result'][j]['id'],
                                        i['annotations'][0]['result'][j]['value']['start'],
                                        i['annotations'][0]['result'][j]['value']['end'],
                                        s,
                                        i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    val_df.to_csv('./data/val_data_append_sentence.csv',index=False)


def train_append_sentence_and_label():
    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
                prev_1_label = '[UNK]'
                prev_2_label = '[UNK]'
            elif(index==1):
                prev_1 = ''
                prev_1_label = '[UNK]'
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                prev_2_label = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                prev_1_label = i['annotations'][0]['result'][j-2]['value']['labels'][0]
                prev_2_label = i['annotations'][0]['result'][j-1]['value']['labels'][0]

            s = prev_1 + '[SEP]' + prev_1_label + '[SEP]' + prev_2 + '[SEP]' + prev_2_label + '[SEP]'  + preprocess(i['annotations'][0]['result'][j]['value']['text'])
            #if(len(j['value']['labels'])>1): print(j['value']['labels'])
            train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                        i['annotations'][0]['result'][j]['value']['start'],
                                        i['annotations'][0]['result'][j]['value']['end'],
                                        s,
                                        i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    train_df.to_csv('./data/train_data_append_sentence_and_label.csv',index=False)

    
def train_append_sentence():
    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
            elif(index==1):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])

            s = prev_1 + '[SEP]' + prev_2 + '[SEP]'  + preprocess(i['annotations'][0]['result'][j]['value']['text'])
            #if(len(j['value']['labels'])>1): print(j['value']['labels'])
            train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                        i['annotations'][0]['result'][j]['value']['start'],
                                        i['annotations'][0]['result'][j]['value']['end'],
                                        s,
                                        i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    train_df.to_csv('./data/train_data_append_sentence.csv',index=False)

def val_append_sentence():
    val_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in val_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
            elif(index==1):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])

            s = prev_1 + '[SEP]' + prev_2 + '[SEP]'  + preprocess(i['annotations'][0]['result'][j]['value']['text'])
            #if(len(j['value']['labels'])>1): print(j['value']['labels'])
            val_df.loc[len(val_df)] = [i['annotations'][0]['result'][j]['id'],
                                        i['annotations'][0]['result'][j]['value']['start'],
                                        i['annotations'][0]['result'][j]['value']['end'],
                                        s,
                                        i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    val_df.to_csv('./data/val_data_append_sentence.csv',index=False)
    
    
#json_to_csv()
#train_prev_labels()
#train_append_labels()
#train_append_sentence()
#val_append_sentence()
#train_append_sentence_and_label()


def train_append_sentence():
    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
                prev_3 = ''            
            elif(index==1):
                prev_1 = ''
                prev_2 = ''
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
            elif(index==2):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-3]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])

            a = preprocess(i['annotations'][0]['result'][j]['value']['text'])
            if(a!=''):
                s = prev_1 + '[SEP]' + prev_2 + '[SEP]'  + prev_3 + '[SEP]' +  a #preprocess(i['annotations'][0]['result'][j]['value']['text'])
                #if(len(j['value']['labels'])>1): print(j['value']['labels'])
                train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                            i['annotations'][0]['result'][j]['value']['start'],
                                            i['annotations'][0]['result'][j]['value']['end'],
                                            s,
                                            i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    train_df.to_csv('./data/train_data_append_sentence_three.csv',index=False)

def val_append_sentence():
    val_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in val_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
                prev_3 = ''            
            elif(index==1):
                prev_1 = ''
                prev_2 = ''
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
            elif(index==2):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-3]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])

            a = preprocess(i['annotations'][0]['result'][j]['value']['text'])
            if(a!=''):
                s = prev_1 + '[SEP]' + prev_2 + '[SEP]'  + prev_3 + '[SEP]' +  a #preprocess(i['annotations'][0]['result'][j]['value']['text'])
                #if(len(j['value']['labels'])>1): print(j['value']['labels'])
                val_df.loc[len(val_df)] = [i['annotations'][0]['result'][j]['id'],
                                            i['annotations'][0]['result'][j]['value']['start'],
                                            i['annotations'][0]['result'][j]['value']['end'],
                                            s,
                                            i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    val_df.to_csv('./data/val_data_append_sentence_three.csv',index=False)

#train_append_sentence()
#val_append_sentence()


def train_append_sentence_prev_labels():
    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
                prev_3 = ''
                pl_1 = '[UNK]'
                pl_2 = '[UNK]'
                pl_3 = '[UNK]'            
            elif(index==1):
                prev_1 = ''
                prev_2 = ''
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = '[UNK]'
                pl_2 = '[UNK]'
                pl_3 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            elif(index==2):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = '[UNK]'
                pl_2 = i['annotations'][0]['result'][j-2]['value']['labels'][0]
                pl_3 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-3]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = i['annotations'][0]['result'][j-3]['value']['labels'][0]
                pl_2 = i['annotations'][0]['result'][j-2]['value']['labels'][0] 
                pl_3 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            
            a = preprocess(i['annotations'][0]['result'][j]['value']['text'])
            if(a!=''):
                s = pl_1 + ' [SEP] ' + pl_2 + ' [SEP] ' +pl_3 + ' [SEP] ' + prev_1 + ' [SEP] ' + prev_2 + ' [SEP] '  + prev_3 + ' [SEP] ' +  a #preprocess(i['annotations'][0]['result'][j]['value']['text'])
                #if(len(j['value']['labels'])>1): print(j['value']['labels'])
                train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                            i['annotations'][0]['result'][j]['value']['start'],
                                            i['annotations'][0]['result'][j]['value']['end'],
                                            s,
                                            i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    train_df.to_csv('./data/train_data_append_sentence_three_append_labels.csv',index=False)

train_append_sentence_prev_labels()

def train_append_sentence_prev_labels_four():
    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
                prev_3 = ''
                prev_4 = ''
                pl_1 = '[UNK]'
                pl_2 = '[UNK]'
                pl_3 = '[UNK]'
                pl_4 = '[UNK]'            
            elif(index==1):
                prev_1 = ''
                prev_2 = ''
                prev_3 = ''
                prev_4 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = '[UNK]'
                pl_2 = '[UNK]'
                pl_3 = '[UNK]'
                pl_4 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            elif(index==2):
                prev_1 = ''
                prev_2 = ''
                prev_3 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_4 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = '[UNK]'
                pl_2 = '[UNK]'
                pl_3 = i['annotations'][0]['result'][j-2]['value']['labels'][0]
                pl_4 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            elif(index==3):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-3]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_4 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = '[UNK]'
                pl_2 = i['annotations'][0]['result'][j-3]['value']['labels'][0]
                pl_3 = i['annotations'][0]['result'][j-2]['value']['labels'][0] 
                pl_4 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-4]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-3]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_4 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = i['annotations'][0]['result'][j-4]['value']['labels'][0]
                pl_2 = i['annotations'][0]['result'][j-3]['value']['labels'][0]
                pl_3 = i['annotations'][0]['result'][j-2]['value']['labels'][0] 
                pl_4 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            a = preprocess(i['annotations'][0]['result'][j]['value']['text'])
            if(a!=''):
                s = pl_1 + '[SEP]' + pl_2 + '[SEP]' + pl_3 + '[SEP]' + pl_4 + '[SEP]' + prev_1 + '[SEP]' + prev_2 + '[SEP]'  + prev_3 + '[SEP]' + prev_4 + '[SEP]' +  a #preprocess(i['annotations'][0]['result'][j]['value']['text'])
                #if(len(j['value']['labels'])>1): print(j['value']['labels'])
                train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                            i['annotations'][0]['result'][j]['value']['start'],
                                            i['annotations'][0]['result'][j]['value']['end'],
                                            s,
                                            i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    train_df.to_csv('./data/train_data_append_sentence_four_append_labels.csv',index=False)

#train_append_sentence_prev_labels_four()


def train_append_sentence_prev_labels_no_sep():
    train_df = pd.DataFrame(columns=['id','start','end','text','label'])
    for i in train_data:
        if(len(i['annotations'])>1): 
            with open('excpetion_in_data.json','w',encoding='utf8') as outfile:
                json.dump(i['annotations'],outfile)
        index = 0
        for j in tqdm(range(len(i['annotations'][0]['result']))):
            #id, start, end, text, label
            if(index == 0):
                prev_1 = ''
                prev_2 = ''
                prev_3 = ''
                pl_1 = '[UNK]'
                pl_2 = '[UNK]'
                pl_3 = '[UNK]'            
            elif(index==1):
                prev_1 = ''
                prev_2 = ''
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = '[UNK]'
                pl_2 = '[UNK]'
                pl_3 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            elif(index==2):
                prev_1 = ''
                prev_2 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = '[UNK]'
                pl_2 = i['annotations'][0]['result'][j-2]['value']['labels'][0]
                pl_3 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            else:
                prev_1 = preprocess(i['annotations'][0]['result'][j-3]['value']['text'])
                prev_2 = preprocess(i['annotations'][0]['result'][j-2]['value']['text'])
                prev_3 = preprocess(i['annotations'][0]['result'][j-1]['value']['text'])
                pl_1 = i['annotations'][0]['result'][j-3]['value']['labels'][0]
                pl_2 = i['annotations'][0]['result'][j-2]['value']['labels'][0] 
                pl_3 = i['annotations'][0]['result'][j-1]['value']['labels'][0]
            
            a = preprocess(i['annotations'][0]['result'][j]['value']['text'])
            if(a!=''):
                s = pl_1 + ' ' + pl_2 + ' ' +pl_3 + ' ' + prev_1 + ' ' + prev_2 + ' '  + prev_3 + ' ' +  a #preprocess(i['annotations'][0]['result'][j]['value']['text'])
                #if(len(j['value']['labels'])>1): print(j['value']['labels'])
                train_df.loc[len(train_df)] = [i['annotations'][0]['result'][j]['id'],
                                            i['annotations'][0]['result'][j]['value']['start'],
                                            i['annotations'][0]['result'][j]['value']['end'],
                                            s,
                                            i['annotations'][0]['result'][j]['value']['labels'][0]]
            index+=1
    train_df.to_csv('./data/train_data_append_sentence_three_append_labels_no_sep.csv',index=False)


#train_append_sentence_prev_labels_no_sep()