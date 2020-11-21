# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
from collections import defaultdict
import json
import re

START_TAG = '<s>'
END_TAG = '</s>'
UNKNOWN_TAG = '<UNK>'
SUM_TAG = '<SUM>'
SEEN_TAG = '<SEEN>'
SUM_WORD = '_SUM'
SEEN_WORD = '_SEEN'
UNK_WORD = '_UNK'
ALL_CAP_WORD = '_ALL_CAP'
NO_CAP_WORD = '_NO_CAP'
OTHER_CAP_WORD = '_OTHER_CAP'
CONTAIN_NUMBER_WORD = '_CONTAIN_NUMBER'
NO_NUMBER_WORD = '_NO_NUMBER'
CONTAIN_DASH_WORD = '_DASH'
NO_DASH_WORD = '_NO_DASH'
CONTAIN_SLASH_WORD = '_SLASH'
NO_SLASH_WORD = '_NO_SLASH'

suffix_list = ["age", "al", "ance", "ence", "dom", "ee", "er", "or", "hood", "ism", "ist", "ity", "ty", "ment", "ness",
               "ry", "ship", "sion", "tion", "xion",
               "able", "ible", "al", "en", "ese", "ful", "i", "ic", "ish", "ive", "ian", "less", "ly", "ous", "y",
               "ate", "en", "ify", "ize", "ise", "ward", "wards", "wise",
               "s", "ed", "ing"]


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    input = open(train_file)
    lines = input.readlines()
    input.close()
    word_tags = {}
    word_count = defaultdict(lambda: 0)
    transition_count = defaultdict(lambda: defaultdict(lambda: 0))
    emission_count = defaultdict(lambda: defaultdict(lambda: 0))

    for line in lines:
        line = line.strip()
        tokens = line.split(' ')
        prev_tag = START_TAG
        for token in tokens:
            word, _, tag = token.rpartition('/')
            if word not in word_tags:
                word_tags[word] = [tag]
            elif tag not in word_tags[word]:
                word_tags[word].append(tag)
            word_count[word] += 1
            transition_count[prev_tag][tag] += 1
            emission_count[tag][word] += 1
            prev_tag = tag
        transition_count[prev_tag][END_TAG] += 1

    # Stats for transition counts
    for tag in transition_count:
        transition_count[tag][SUM_TAG] = sum(transition_count[tag].values())
        transition_count[tag][SEEN_TAG] = len(transition_count[tag])

    # Collect words that appear only once
    unusual_words = set()
    for word in word_count:
        if word_count[word] == 1:
            unusual_words.add(word)

    for tag in emission_count:
        suffix_counts = {}
        temp_dict = {'sum': 0, 'seen': 0, 'unk': 0, 'all_cap': 0, 'no_cap': 0, 'other_cap': 0,
                     'contain_number': 0, 'contain_slash': 0, 'contain_dash': 0}
        for word in emission_count[tag]:
            cur_count = emission_count[tag][word]
            temp_dict['sum'] += cur_count
            temp_dict['seen'] += 1
            if word in unusual_words:
                temp_dict['unk'] += 1
            if word.isupper():
                temp_dict['all_cap'] += cur_count
            elif word.islower():
                temp_dict['no_cap'] += cur_count
            else:
                temp_dict['other_cap'] += cur_count
            if bool(re.search(r'\d', word)):
                temp_dict['contain_number'] += cur_count
            if '-' in word:
                temp_dict['contain_dash'] += cur_count
            if '/' in word:
                temp_dict['contain_slash'] += cur_count
            for suffix in suffix_list:
                if word.lower().endswith(suffix):
                    if suffix in suffix_counts:
                        suffix_counts[suffix] += 1
                    else:
                        suffix_counts[suffix] = 1
        emission_count[tag][SUM_WORD] = temp_dict['sum']
        emission_count[tag][SEEN_WORD] = temp_dict['seen']
        emission_count[tag][UNK_WORD] = temp_dict['unk']
        emission_count[tag][ALL_CAP_WORD] = temp_dict['all_cap']
        emission_count[tag][NO_CAP_WORD] = temp_dict['no_cap']
        emission_count[tag][OTHER_CAP_WORD] = temp_dict['other_cap']
        emission_count[tag][CONTAIN_NUMBER_WORD] = temp_dict['contain_number']
        emission_count[tag][CONTAIN_DASH_WORD] = temp_dict['contain_dash']
        emission_count[tag][CONTAIN_SLASH_WORD] = temp_dict['contain_slash']
        emission_count[tag][NO_NUMBER_WORD] = temp_dict['sum'] - temp_dict['contain_number']
        emission_count[tag][NO_DASH_WORD] = temp_dict['sum'] - temp_dict['contain_dash']
        emission_count[tag][NO_SLASH_WORD] = temp_dict['sum'] - temp_dict['contain_slash']
        for suffix in suffix_list:
            if suffix in suffix_counts:
                emission_count[tag]["_ENDS_WITH" + suffix] = suffix_counts[suffix]
            else:
                emission_count[tag]["_ENDS_WITH" + suffix] = 0

    # Preprocess counts to logarithm
    for tag in transition_count:
        for next_tag in transition_count[tag]:
            cur = transition_count[tag][next_tag]
            transition_count[tag][next_tag] = math.log(cur) if cur != 0 else -math.inf
    for tag in transition_count:
        if tag == START_TAG:
            continue
        for word in emission_count[tag]:
            cur = emission_count[tag][word]
            emission_count[tag][word] = math.log(cur) if cur != 0 else -math.inf

    model = {'word_tags': word_tags, 'word_count': word_count, 'transition_count': transition_count,
             'emission_count': emission_count}

    with open(model_file, 'w') as f:
        json.dump(model, f, sort_keys=True)
        f.close()

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
