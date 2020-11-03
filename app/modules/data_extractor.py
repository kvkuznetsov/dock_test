import pandas as pd
import re
import requests
from modules.get_stand_repls import *

# file = open('extractor_log.txt', 'w', encoding='utf-8')

# from pymorphy2.tokenizers import simple_word_tokenize
# from enchant import Dict
from nltk.tokenize import sent_tokenize
from pymorphy2.tokenizers import simple_word_tokenize
from itertools import groupby

from numpy import isnan
from datetime import datetime

# def spell_check(text, checker):
#     outp = ''
#     text = simple_word_tokenize(text)
#     for idx, word in enumerate(text):
#         if checker.check(word):
#             if idx and word not in (";!.,"):
#                 outp += ' '
#             outp += word
#         else:
#             suggestions = checker.suggest(word)
#             if suggestions:
#                 sugg = suggestions[0]
#                 if idx and sugg not in (";!.,"):
#                     outp += ' '
#                 outp += sugg
#     ##print(outp)
#     return 

def split_contexts(text, split_json):
    outp = []
    if type(text) == str:
        outp.append(text)
        if "nltk" in split_json:
            if split_json["nltk"]:
                outp = [context for fragment in outp for context in sent_tokenize(fragment)]
        if "split_local" in split_json:
            for char in split_json["split_local"]:
                outp = [context for fragment in outp for context in fragment.split(char)]
        if "filter" in split_json:
            if "regex_filter" in split_json["filter"]:
                filtered_output = []
                reg = '|'.join(regex for regex in split_json["filter"]["regex_filter"])
                for context_id, context in enumerate(outp):
                    matches = re.finditer(reg, context)
                    if matches:
                        prev_border = 0
                        for match_id, match in enumerate(matches):
                            left_match_border, right_match_border = match.span()
                            left_context = simple_word_tokenize(context[prev_border:left_match_border])
                            if "left_from_match" in split_json["filter"]["cut_text"]:
                                left_words = split_json["filter"]["cut_text"]["left_from_match"]
                                left_context = left_context[-left_words:]
                            phrase = ' '.join(left_context) + ' ' + match.group(0)
                            # phrase = ' '.join(left_context) + match.group(0)
                            filtered_output.append(phrase)
                            prev_border = right_match_border
                outp = filtered_output
    return outp

def filter_classified(responses, filters):
    if filters:
        outp = []
        for resp in responses:
            if resp['predicted'] in filters:
                outp.append(resp['text'])
        return outp
    else:
        return [(i['text'], i['predicted']) for i in responses]


class DataExtractor():
    def __init__(self, json):
        self.error = {}
        self.df = {}
        self.res_json = []
        self.json = json
        if 'iter_lst' in self.json:
            self.iter_lst = self.json['iter_lst']
        else:
            self.iter_lst = []
        if 'modf_text' in self.json:
            if 'spell_check' in self.json:
                if self.json['modf_text']['spell_check']:
                    ## requires RU dict and aff files in lib/enchant:
                    self.checker = Dict("ru_RU")

    def check_json(self):
        '''Проверяет Json на отсуствующие ключи'''
        if 'task' not in self.json:
            self.error = {"error": "wrong json",
                          'detail': 'json without key "task"'}
        if 'type' not in self.json:
            self.error = {"error": "wrong json",
                          'detail': 'json without key "type"'}
        if 'regex' not in self.json:
            self.error = {"error": "wrong json",
                          'detail': 'json without key "regex"'}
        if 'data' not in self.json:
            self.error = {"error": "wrong json",
                          'detail': 'json without key "data"'}

    def create_df(self):
        '''Создает датафрейм и изменяет текст'''
        def make_repl(row, repls, regex=False):
            if regex:
                for key, val in repls.items():
                    row = str(row)
                    row = re.sub(key, val, row)
            else:
                for key, val in repls.items():
                    row = str(row)
                    row = row.replace(key, val)
            row = row.replace(chr(776), '')
            return row

        self.df = pd.DataFrame(data=self.json['data'])
        # print(self.df.shape)
        self.df['value'] = None
        self.df['contexts'] = None 
        if 'modf_text' in self.json:
            if 'low_case' in self.json['modf_text']:
                if self.json['modf_text']['low_case']:
                    self.df['text'] = self.df['text'].str.lower()
            if 'replace' in self.json['modf_text']:
                repls = self.json['modf_text']['replace']
                if 'standart' in repls:
                    repl_stand = repls['standart']
                    if 'eng_rus' in repl_stand:
                        self.df['text'] = self.df['text'].\
                                          apply(lambda x: \
                                                make_repl(x, repl_eng_rus()))
                    if 'str_int_to_int' in repl_stand:
                        self.df['text'] = self.df['text'].\
                                          apply(lambda x: \
                                                make_repl(x, repl_str_int(), regex=True))
                    if 'trim' in repl_stand:
                        self.df['text'] = self.df['text'].\
                                          apply(lambda x: \
                                                re.sub('\\s+', ' ', x))

                if 'regex_replace' in repls:
                    for repl in repls['regex_replace']:
                        regs = repl['old']
                        for reg in regs:
                            ## temporary:
                            # for text in self.df['text']:
                            #     match = re.search(reg, text)
                            #     if match:
                            #         #print(match.group(0))
                            #print(reg)
                            try:
                                re.compile(reg)
                            except re.error:
                                pass
                                #print(f"Bad regexp - {reg}")
                            # print('old: '+reg+'\nnew: '+repl['new']+'\ntext: '+self.df['text'][0], end='\n\n', file=file)
                            self.df['text'] = self.df['text'].\
                                              apply(lambda x:\
                                                    re.sub(reg, repl['new'], x))
                if 'local' in repls:
                    for repl in repls['local']:
                        repl_old = repl['old']
                        for old in repl_old:
                            self.df['text'] = \
                            self.df['text'].str.replace(old, repl['new'])
            
            # if 'spell_check' in self.json['modf_text']:
            #     if self.json['modf_text']['spell_check']:
            #         #print("Spellchecking ... ")
            #         self.df['text'] = self.df['text'].apply(lambda x: spell_check(x, self.checker))
    
    def split_text(self):
        # print("stripping contexts: ", datetime.now())
        self.df['contexts'] = self.df['text'].apply(lambda x: split_contexts(x, self.json["split_text"]))

    def classify_text(self, api_params):

        json_to_send = self.json['classify_text']['json']
        json_to_send['data'] = [{"id": text_id, "text": context} for text_id, contexts in zip(self.df['id'], self.df['contexts']) for context in contexts]

        # api_params = self.json['classify_text']['api']
        url = api_params["host"]
        if "port" in api_params:
            url += ':' + str(api_params["port"])
        if "url_method" in api_params:
            url += '/'+api_params["url_method"]
                
        # print("sending data to classifier ... ", datetime.now())

        if json_to_send["data"]:
            if api_params["method"] == "POST":
                resp = requests.post(url, json=json_to_send)
            resp_json = resp.json()

            # print('classified data recieved', datetime.now())

            resp_json = {k: list(v) for k,v in groupby(resp_json, key = lambda x: x['id'])}

        else:
            ## В случае, если в выборке нет подходящих контекстов
            resp_json = ()

        # print('filtering classified data', datetime.now())

        if 'classify_filter' in self.json['classify_text']:
            clf_filter = self.json['classify_text']['classify_filter']
        else:
            clf_filter = []
        
        self.df['raw_contexts'] = self.df['contexts']
        self.df['contexts'] = self.df.apply(lambda x: filter_classified(resp_json[x['id']], clf_filter) if x['id'] in resp_json else [],
        axis=1)

        # print('classified data filtered', datetime.now())

    def regex_val(self):
        '''Использовать регулярное выражение'''
        def reg(value, text, reg_flt, reg_fnd, reg_neg, multi, name):
            '''Если задана регулярка фильтрации использовать её сначала
               Используется для get_float_from_text'''
            if reg_neg:
                if re.search(reg_neg, text):
                    return value
            if (value is None or value == [] or value == [[], None]):
                # print(text)
                if not reg_flt == '':
                    lst_x = [m.group() for m in re.finditer(reg_flt, text)]
                    x = []
                    for elm in lst_x:
                        el = re.findall(reg_fnd, elm)
                        for e in el:
                            if type(e) == tuple:
                                for et in e:
                                    et = float(et.replace(',', '.'))
                                    et = et * multi
                                    if et not in x:
                                        x.append(et)
                            else:
                                e = float(e.replace(',', '.')) * multi
                                if e not in x:
                                    x.append(e)
                else:
                    x = re.findall(reg_fnd, text)

                if x is not None and not x == []:
                    res = [x, reg_fnd, name]
                else:
                    res = [x, None]
                return res
            else:
                return value

        def reg_many_float(value, text, reg_flt, reg_fnd, reg_neg, multi, name, iter_lst, min_val=None, max_val=None):
            '''Если задана регулярка фильтрации использовать её сначала
               Используется для get_many_float_from_text'''
            check_min_val, check_max_val = False, False
            if min_val is not None:
                check_min_val = True
            if max_val is not None:
                check_max_val = True
            
            if reg_neg:
                if re.search(reg_neg, text):
                    return value
            if iter_lst == []:
                if value is None or value == [] or value == [[], None]:
                    value = []
                x = []
                if reg_flt:
                    lst_x = [i.group() for i in re.finditer(reg_flt, text)]
                    for elm in lst_x:
                        for match in re.findall(reg_fnd, elm):
                            match = float(match.replace(',','.'))
                            match = match * multi
                            if check_min_val:
                                if match < min_val:
                                    continue
                            if check_max_val:
                                if match > max_val:
                                    continue
                            x.append(match)
                else:
                    lst_x = re.findall(reg_fnd, text)
                    for elm in lst_x:
                        elm = float(elm.replace(',', '.'))
                        elm = elm * multi
                        if check_min_val:
                            if elm < min_val:
                                continue
                        if check_max_val:
                            if elm > max_val:
                                continue
                        x.append(elm)

                if x is not None and not x == []:
                    if iter_lst == []:
                        res = [x, reg_fnd, name]
                    value.append(res)
                return value

            else:
                # для списка значений. например мкад, кад итд.
                for elm_ in iter_lst:
                    if value is None or value == [] or value == [[], None]:
                        value = []
                    x = []
                    reg_dnc = reg_fnd.replace('_ELEM_', elm_)
                    lst_x = re.findall(reg_dnc, text)
                    for elm in lst_x:
                        elm = float(elm.replace(',', '.'))
                        elm = elm * multi
                        if check_min_val:
                            if elm < min_val:
                                continue
                        if check_max_val:
                            if elm > max_val:
                                continue
                        x.append(elm)

                    if x is not None and not x == []:
                        res = [x, reg_dnc, elm_]
                        value.append(res)
                return value


        def reg_spr(value, text, reg_fnd, name, reg_neg):
            '''Используется для get_str_from_text'''
            if value is None or value == [] or value == [[], None]:
                x_lst = []
                reg_lst = []
            else:
                x_lst = value[0]
                reg_lst = value[1]
            x = re.search(reg_fnd, text)
            #if x:
                ##print(text, x)

            if not reg_neg == '':
                neg = re.findall(reg_neg, text)
                if len(neg):
                    x = None

            if x is not None:
                if name not in x_lst:
                    x_lst.append(name)
                    reg_lst.append(reg_fnd)

            res = [x_lst, reg_lst]

            return res


        def reg_str(value, text, reg_fnd, name, reg_neg):
            '''Используется для get_str_from_text'''
            if value is None or value == [] or value == [[], None]:
                x_lst = []
                reg_lst = []
            else:
                x_lst = value[0]
                reg_lst = value[1]
            x = re.search(reg_fnd, text)
            match = re.findall(reg_fnd, text)
            #if x:
                ##print(text, x)

            if not reg_neg == '':
                neg = re.findall(reg_neg, text)
                if len(neg):
                    x = None

            if x is not None:
                if name not in x_lst:
                    x_lst.append(match)
                    reg_lst.append(reg_fnd)

            res = [x_lst, reg_lst]

            return res
        def reg_naz(value, text, reg_flt, reg_fnd, name, group, podsegm, reg_neg):
            '''Используется для get_nazn_from_text'''
            if value is None:
                value = []

            if not reg_neg == '':
                neg = re.findall(reg_neg, text)
                if len(neg):
                    return value

            if not reg_flt == '':
                tpl_x = re.findall(reg_flt, text)
                for lst in tpl_x:
                    if not type(lst) == tuple:
                        lst = (lst,)
                    for elm in lst:
                        if len(elm):
                            fnd = re.findall(reg_fnd, elm)
                            if len(fnd):
                                res = {
                                    "name": name,
                                    "group": group,
                                    "podsegm": podsegm,
                                    "reg_fnd": reg_fnd
                                }
                                if res not in value:
                                    value.append(res)
            return value
        
        def regex_classified(value, contexts, reg_fnd, reg_neg, multi, two_d, name, diap, sum_values, rm_dupl=False):
            '''Если задана регулярка фильтрации использовать её сначала
               Используется для get_float_with_context'''
            
            if not value:
                value = [[],[],[]]
            lst_x = []
            for context in contexts:
                if not reg_neg or not re.findall(reg_neg, context):
                    lst_x = re.findall(reg_fnd, context)
                    # x = []
                    if two_d:
                        for e in lst_x:
                            float1 = e[0]
                            float2 = e[-1]
                            float1 = float(float1.replace(',','.'))
                            float2 = float(float2.replace(',','.'))
                            area = float1 * float2 * multi**2
                            if diap:
                                if area >= diap[0] and area <= diap[1]:
                                    pass
                                else:
                                    ## Thou shall not pass!
                                    continue
                            if rm_dupl:
                                if area not in value[0]:
                                    value[0].append(area)
                                    if reg_fnd not in value[1]:
                                        value[1].append(reg_fnd)
                                    if name not in value[2]:
                                        value[2].append(name)
                            else:
                                value[0].append(area)
                                value[1].append(reg_fnd)
                                value[2].append(name)
                    else:
                        for e in lst_x:
                            if type(e) == tuple:
                                if sum_values:
                                    val = 0
                                for et in e:
                                    try:  
                                        et = float(et.replace(',', '.'))
                                        et *= multi
                                        
                                        if sum_values:
                                            val += et

                                        if diap:
                                            if et >= diap[0] and et <= diap[1]:
                                                pass
                                            else:
                                                continue
                                        
                                        if not sum_values:
                                            if rm_dupl:
                                                if et not in value[0]:
                                                    # x.append(et)
                                                    value[0].append(et)
                                                    if reg_fnd not in value[1]:
                                                        value[1].append(reg_fnd)
                                                    if name not in value[2]:
                                                        value[2].append(name)
                                            else:
                                                value[0].append(et)
                                                value[1].append(reg_fnd)
                                                value[2].append(name)
                                    except:
                                        continue
                                if sum_values:
                                    if val:
                                        if diap:
                                            if val >= diap[0] and val <= diap[1]:
                                                pass
                                            else:
                                                continue
                                        if rm_dupl:
                                            if val not in value[0]:
                                                # x.append(et)
                                                value[0].append(val)
                                                if reg_fnd not in value[1]:
                                                    value[1].append(reg_fnd)
                                                if name not in value[2]:
                                                    value[2].append(name)
                                        else:
                                            value[0].append(val)
                                            value[1].append(reg_fnd)
                                            value[2].append(name)
                            else:
                                e = float(e.replace(',', '.')) * multi
                                if diap:
                                    if e >= diap[0] and e <= diap[1]:
                                        pass
                                    else:
                                        continue
                                if rm_dupl:
                                    if e not in value[0]:
                                        # x.append(e)
                                        value[0].append(e)
                                        if reg_fnd not in value[1]:
                                            value[1].append(reg_fnd)
                                        if name not in value[2]:
                                            value[2].append(name)
                                else:
                                    value[0].append(e)
                                    value[1].append(reg_fnd)
                                    value[2].append(name)
            return value

        if not len(self.json['regex']):
            self.error = {"error": "wrong json",
                          'detail': 'empty regex list'}
            return
        if self.json['type'] == "get_float_from_text":
            for reg_dict in self.json['regex']:
                if 'regex_filter' not in reg_dict:
                    reg_dict['regex_filter'] = ''
                if 'regex_neg' not in reg_dict:
                    reg_dict['regex_neg'] = None
                if 'multi' not in reg_dict:
                    reg_dict['multi'] = 1

                self.df['value'] = \
                self.df.apply(lambda x: reg(x.value,
                                            x.text,
                                            reg_dict['regex_filter'],
                                            reg_dict['regex_find'],
                                            reg_dict['regex_neg'],
                                            reg_dict['multi'],
                                            reg_dict['name']),
                                            axis=1)
        if self.json['type'] == "get_many_float_from_text":
            for reg_dict in self.json['regex']:
                if 'regex_filter' not in reg_dict:
                    reg_dict['regex_filter'] = ''
                if 'regex_neg' not in reg_dict:
                    reg_dict['regex_neg'] = ''
                if 'multi' not in reg_dict:
                    reg_dict['multi'] = 1
                if 'min' not in reg_dict:
                    reg_dict['min'] = None
                if 'max' not in reg_dict:
                    reg_dict['max'] = None
                self.df['value'] = \
                self.df.apply(lambda x: reg_many_float(x.value,
                                                       x.text,
                                                       reg_dict['regex_filter'],
                                                       reg_dict['regex_find'],
                                                       reg_dict['regex_neg'],
                                                       reg_dict['multi'],
                                                       reg_dict['name'],
                                                       self.iter_lst,
                                                       min_val=reg_dict["min"],
                                                       max_val=reg_dict["max"]),
                                                       axis=1)
        elif self.json['type'] == "get_str_from_text":
            only_shortest = False
            if "modf_text" in self.json:
                if "only_shortest_str" in self.json['modf_text']:
                    if self.json['modf_text']['only_shortest_str']:
                        only_shortest = True
            ## чтобы в датафрейм записывался наименьший по длине метч:
            if only_shortest:
                self.df['value'] =\
                self.df.apply(lambda x: self.apply_regex(x.text, x.value), axis=1)
                # if not "value" in self.df:
                #     self.df["value"] = [[] for i in range(self.df.shape[1])]
                # for index, row in self.df.iterrows():
                #     self.df.at[index, 'value'] = self.apply_regex(row['text'], row['value'])
            else:
                for reg_dict in self.json['regex']:
                    # try:
                    #     re.compile(reg_dict['regex_find'])
                    # except:
                    #     print(f"Bad regexp - {reg_dict['regex_find']}")
                    
                    if 'regex_filter' not in reg_dict:
                        # try:
                        #     re.compile(reg_dict['regex_filter'])
                        # except:
                        #     print(f"Bad regexp - {reg_dict['regex_filter']}")
                        
                        reg_dict['regex_filter'] = ''
                    if 'regex_neg' not in reg_dict:
                        # try:
                        #     re.compile(reg_dict['regex_neg'])
                        # except:
                        #     print(f"Bad regexp - {reg_dict['regex_neg']}")
                        
                        reg_dict['regex_neg'] = ''
                    self.df['value'] = \
                    self.df.apply(lambda x: reg_spr(x.value,
                                                    x.text,
                                                    reg_dict['regex_find'],
                                                    reg_dict['name'],
                                                    reg_dict['regex_neg']),
                                                    axis=1)


        elif self.json['type'] == "get_str_match_from_text":
            for reg_dict in self.json['regex']:
                if 'regex_filter' not in reg_dict:
                    reg_dict['regex_filter'] = ''
                if 'regex_neg' not in reg_dict:
                    reg_dict['regex_neg'] = ''
                self.df['value'] = \
                self.df.apply(lambda x: reg_str(x.value,
                                                x.text,
                                                reg_dict['regex_find'],
                                                reg_dict['name'],
                                                reg_dict['regex_neg']),
                                                axis=1)

        elif self.json['type'] == "get_nazn_from_text":
            for reg_dict in self.json['regex']:
                if 'regex_filter' not in reg_dict:
                    reg_dict['regex_filter'] = ''
                if 'regex_neg' not in reg_dict:
                    reg_dict['regex_neg'] = ''
                self.df['value'] = \
                self.df.apply(lambda x: reg_naz(x.value,
                                                x.text,
                                                reg_dict['regex_filter'],
                                                reg_dict['regex_find'],
                                                reg_dict['name'],
                                                reg_dict['group'],
                                                reg_dict['podsegm'],
                                                reg_dict['regex_neg']),
                                                axis=1)
        
        elif self.json['type'] == 'get_float_with_context':
            for reg_dict in self.json['regex']:
                if 'regex_filter' not in reg_dict:
                    reg_dict['regex_filter'] = ''
                if 'regex_neg' not in reg_dict:
                    reg_dict['regex_neg'] = ''
                if "_2d" not in reg_dict:
                    reg_dict["_2d"] = False
                
                if "multi" not in reg_dict:
                    reg_dict['multi'] = 1
                
                if "diap" in reg_dict:
                    diap = (reg_dict["diap"]["lower_value"], reg_dict["diap"]["upper_value"])
                else:
                    diap = None
                
                if "sum_values" not in reg_dict:
                    reg_dict["sum_values"] = False
                
                self.df['value'] = \
                self.df.apply(lambda x: regex_classified(x['value'],
                                                x['contexts'],
                                                reg_dict['regex_find'],
                                                reg_dict['regex_neg'],
                                                reg_dict['multi'],
                                                reg_dict['_2d'],
                                                reg_dict['name'],
                                                diap,
                                                reg_dict["sum_values"]),
                                            axis=1)

    def apply_regex(self, text, value):
        '''Используется для get_str_from_text'''

        # print(text)
        if value is None or value == [] or value == [[], None]:
            x_lst = []
            reg_lst = []
        else:
            x_lst = value[0]
            reg_lst = value[1]
        
        spans1 = []
        spans2 = []

        for reg_dict in self.json["regex"]:
            # x = re.search(reg_dict["regex_find"], text)
            ## to get all matches and not just the first:

            # try:
            #     re.compile(reg_dict['regex_find'])
            # except:
            #     print(f"Bad regexp - {reg_dict['regex_find']}")

            x = list(re.finditer(reg_dict["regex_find"], text))
            # print([i for i in x], reg_dict["name"])
            no_neg = True
            if "regex_neg" in reg_dict:

                # # try:
                # #     re.compile(reg_dict['regex_neg'])
                # # except:
                # #     print(f"Bad regexp - {reg_dict['regex_neg']}")

                neg_matches = re.findall(reg_dict["regex_neg"], text)
                if len(neg_matches):
                    #print(reg_dict["name"])
                    #print("Neg matches: ")
                    # print(re.search(reg_dict["regex_neg"], text).group(0))
                    no_neg = False
                    #print("End of neg matches")
            if no_neg and x:
                #print("adding category - ", reg_dict["name"])
                if "only_unclassified" in reg_dict:
                    if reg_dict["only_unclassified"]:
                        spans2 += [(y, reg_dict["name"], reg_dict["regex_find"]) for y in x]
                        continue
                spans1 += [(y, reg_dict["name"], reg_dict["regex_find"]) for y in x]

        ## O(nlogn):  
        spans1 = sorted(spans1, key=lambda x: x[0].span())
        #print(spans1)

        prev_start_index = -1

        for i in range(len(spans1)):
            match, name, regex = spans1[i]
            span = match.span()
            #print(match, match.group(0))
            # to do:
            # также сделать с негативными регулярками - 
            # если есть негативная регулярка и позитивная с одинаковым индексом стартового символа,
            # то они сравниваются по длине
            # и срабатывает та, которая короче:
            if span[0] != prev_start_index:
                #print(prev_start_index, match, match.group(0))
                prev_start_index = span[0]
                if name not in x_lst:
                    x_lst.append(name)
                    reg_lst.append(regex)
        spans2 = sorted(spans2, key=lambda x: x[0].span())
        #print(spans1)

        # print(spans1, spans2)

        prev_start_index = -1

        for i in range(len(spans2)):
            match, name, regex = spans2[i]
            span = match.span()
            #print(match, match.group(0))
            # to do:
            # также сделать с негативными регулярками - 
            # если есть негативная регулярка и позитивная с одинаковым индексом стартового символа,
            # то они сравниваются по длине
            # и срабатывает та, которая короче:
            if span[0] != prev_start_index:
                #print(prev_start_index, match, match.group(0))
                prev_start_index = span[0]
                if not x_lst:
                    x_lst.append(name)
                    reg_lst.append(regex)
                    break
        res = (x_lst, reg_lst)
        return res

    def get_res_json(self):
        '''Получить итоговый Json'''
        res_list = []
        # print(self.df.shape)
        for row in self.df.iterrows():
            dict_elem = row[1].to_dict()
            res_dict = {}
            res_dict['id'] = dict_elem['id']

            ## для дебага, закомментировать когда не будет нужно:
            # res_dict['text with replacements'] = dict_elem['text']

            for key in dict_elem:
                if key[0] == '_':
                    if type(dict_elem[key]) == float:
                        if not isnan(dict_elem[key]):
                            res_dict[key] = dict_elem[key]
                    else:
                        res_dict[key] = dict_elem[key]
            if self.json['type'] == "get_float_from_text":
                # print(dict_elem['text'])
                res_dict['value'] = []
                for x in dict_elem['value'][0]:
                    if 'diap' in self.json['regex'][0]:
                        if type(x) == str:
                            x = float(x.replace(',', '.'))
                        low = self.json['regex'][0]['diap']['lower_value']
                        upp = self.json['regex'][0]['diap']['upper_value']
                        if x > low and x < upp:
                            res_dict['value'].append(x)
                    else:
                        res_dict['value'].append(x)
                res_dict['regex'] = dict_elem['value'][1]
                try:
                    res_dict['name'] = dict_elem['value'][2]
                except Exception:
                    res_dict['name'] = ''
            elif self.json['type'] == "get_many_float_from_text":
                res_lst = []
                for x in dict_elem['value']:
                    res_dct = {}
                    res_dct['value'] = x[0]
                    res_dct['regex'] = x[1]
                    res_dct['name'] = x[2]
                    res_lst.append(res_dct)
                res_dict['id'] = dict_elem['id']
                res_dict['result'] = res_lst
            elif self.json['type'] == "get_str_from_text":
                res_dict['value'] = dict_elem['value'][0]
                res_dict['regex'] = dict_elem['value'][1]
            elif self.json['type'] == "get_str_match_from_text":
                res_dict['value'] = dict_elem['value'][0]
                res_dict['regex'] = dict_elem['value'][1]
            elif self.json['type'] == "get_nazn_from_text":
                res_dict['value'] = dict_elem['value']
            elif self.json['type'] == "get_float_with_context":
                res_dict["contexts"] = dict_elem["contexts"]
                #res_dict["raw_contexts"] = dict_elem["raw_contexts"]
                res_dict["value"] = dict_elem["value"][0]
                res_dict['regex'] = dict_elem['value'][1]
                #res_dict['text with replacements'] = dict_elem['text']
                try:
                    res_dict['name'] = dict_elem['value'][2]
                except Exception:
                    res_dict['name'] = ''
            res_list.append(res_dict)
        self.res_json = res_list
