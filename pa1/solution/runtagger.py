# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import re
import json

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


def viterbi(text, model):
    word_tags = model['word_tags']
    word_count = model['word_count']
    transition_count = model['transition_count']
    emission_count = model['emission_count']
    tags = set(emission_count.keys())
    all_words = set(word_count.keys())

    words = text.strip().split(' ')
    V = [{} for _ in range(len(words))]
    prev_tag_set = []

    # First word
    for tag in transition_count[START_TAG]:
        if tag in [SUM_TAG, SEEN_TAG]:
            continue
        trans = transition_count[START_TAG][tag] - transition_count[START_TAG][SUM_TAG]
        word = words[0]
        if word not in all_words:
            if word.lower() in all_words:
                word = word.lower()
            elif word.upper() in all_words:
                word = word.upper()
        if word not in all_words:
            _sum = emission_count[tag][SUM_WORD]
            if word.islower():
                emi = emission_count[tag][UNK_WORD] + emission_count[tag][NO_CAP_WORD] - 2 * _sum
            elif word.isupper():
                emi = emission_count[tag][UNK_WORD] + emission_count[tag][ALL_CAP_WORD] - 2 * _sum
            else:
                emi = emission_count[tag][UNK_WORD] + emission_count[tag][OTHER_CAP_WORD] - 2 * _sum
            if bool(re.search(r'\d', word)):
                emi += emission_count[tag][CONTAIN_NUMBER_WORD] - _sum
            else:
                emi += emission_count[tag][NO_NUMBER_WORD] - _sum
            if '-' in word:
                emi += emission_count[tag][CONTAIN_DASH_WORD] - _sum
            else:
                emi += emission_count[tag][NO_DASH_WORD] - _sum
            if '/' in word:
                emi += emission_count[tag][CONTAIN_SLASH_WORD] - _sum
            else:
                emi += emission_count[tag][NO_SLASH_WORD] - _sum
            for suffix in suffix_list:
                if word.lower().endswith(suffix):
                    emi += emission_count[tag]["_ENDS_WITH" + suffix] - _sum
        elif word not in emission_count[tag]:
            emi = -math.inf
        else:
            emi = emission_count[tag][word] - emission_count[tag][SUM_WORD]
        V[0][tag] = [trans + emi, START_TAG]
        if trans + emi != -math.inf and tag != END_TAG:  # ???
            prev_tag_set.append(tag)

    # Remaining words
    for i in range(1, len(words)):
        word = words[i]
        if word not in all_words:
            if word.lower() in all_words:
                word = word.lower()
            elif word.upper() in all_words:
                word = word.upper()
        cur_tag_set = []
        if word not in all_words:
            for tag in tags:
                _sum = emission_count[tag][SUM_WORD]
                if word.islower():
                    emi = emission_count[tag][UNK_WORD] + emission_count[tag][NO_CAP_WORD] - 2 * _sum
                elif word.isupper():
                    emi = emission_count[tag][UNK_WORD] + emission_count[tag][ALL_CAP_WORD] - 2 * _sum
                else:
                    emi = emission_count[tag][UNK_WORD] + emission_count[tag][OTHER_CAP_WORD] - 2 * _sum
                if bool(re.search(r'\d', word)):
                    emi += emission_count[tag][CONTAIN_NUMBER_WORD] - _sum
                else:
                    emi += emission_count[tag][NO_NUMBER_WORD] - _sum
                if '-' in word:
                    emi += emission_count[tag][CONTAIN_DASH_WORD] - _sum
                else:
                    emi += emission_count[tag][NO_DASH_WORD] - _sum
                if '/' in word:
                    emi += emission_count[tag][CONTAIN_SLASH_WORD] - _sum
                else:
                    emi += emission_count[tag][NO_SLASH_WORD] - _sum
                for suffix in suffix_list:
                    if word.lower().endswith(suffix):
                        emi += emission_count[tag]["_ENDS_WITH" + suffix] - _sum
                for prev_tag in prev_tag_set:
                    trans = transition_count[prev_tag][tag] - transition_count[prev_tag][SUM_TAG] \
                        if tag in transition_count[prev_tag] else -math.inf
                    cur_val = emi + trans + V[i - 1][prev_tag][0]
                    if tag not in V[i]:
                        V[i][tag] = [cur_val, prev_tag]
                    elif cur_val > V[i][tag][0]:
                        V[i][tag] = [cur_val, prev_tag]
                if V[i][tag][0] != -math.inf and tag != END_TAG:  # ???
                    cur_tag_set.append(tag)
        else:
            for tag in word_tags[word]:
                emi = emission_count[tag][word] - emission_count[tag][SUM_WORD]
                for prev_tag in prev_tag_set:
                    trans = transition_count[prev_tag][tag] - transition_count[prev_tag][SUM_TAG] if tag in \
                                                                                                     transition_count[
                                                                                                         prev_tag] else - \
                    transition_count[prev_tag][SUM_TAG]  # ??????
                    cur_val = emi + trans + V[i - 1][prev_tag][0]
                    if tag not in V[i]:
                        V[i][tag] = [cur_val, prev_tag]
                    elif cur_val > V[i][tag][0]:
                        V[i][tag] = [cur_val, prev_tag]
                if V[i][tag][0] != -math.inf and tag != END_TAG:  # ???
                    cur_tag_set.append(tag)
        prev_tag_set = cur_tag_set
    best = None
    for prev_tag in prev_tag_set:
        if END_TAG in transition_count[prev_tag]:
            trans = transition_count[prev_tag][END_TAG] - transition_count[prev_tag][SUM_TAG]
        else:
            trans = -math.inf
        cur = trans + V[len(words) - 1][prev_tag][0]
        best = [cur, prev_tag] if best is None or cur > best[0] else best
    backtrack = best[1]
    pairs = ["" for _ in range(len(words))]
    i = len(words) - 1
    while i >= 0:
        pairs[i] = backtrack
        backtrack = V[i][backtrack][1]
        i -= 1
    res = ""
    for i in range(len(words)):
        res += words[i] + '/' + pairs[i] + ' '
    return res


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    with open(test_file, 'r') as f:
        lines = f.readlines()
        f.close()
    with open(model_file, 'r') as f:
        model = json.load(f)
        f.close()
    ans = ""
    for line in lines:
        ans += viterbi(line, model) + '\n'
    with open(out_file, 'w') as f:
        f.write(ans)
        f.close()
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
