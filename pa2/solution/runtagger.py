# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import sys
import torch
from buildtagger import POSTagger, WordEncoder, CharEncoder
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tag_sentence(test_file, model_file, out_file):
    with open(test_file, 'r') as f:
        lines = f.readlines()
        f.close()

    # Hyperparameters
    WORD_EMBEDDING_DIM = 100
    CHAR_EMBEDDING_DIM = 50
    CHAR_CNN_FILTER_SIZE = 3
    HIDDEN_DIM = 128
    N_LAYERS = 2
    DROPOUT = 0.25

    word_dict, tag_dict, char_dict, model_state_dict = torch.load(model_file)
    ix_to_tag = {}
    for tag in tag_dict:
        ix_to_tag[tag_dict[tag]] = tag

    # Computed values
    WORD_INPUT_DIM = len(word_dict) + 1
    WORD_PAD_IDX = len(word_dict)
    CHAR_INPUT_DIM = len(char_dict) + 1
    CHAR_PAD_IDX = len(char_dict)
    OUTPUT_DIM = len(tag_dict) + 1

    word_encoder = WordEncoder(vocab_size=WORD_INPUT_DIM, embedding_dim=WORD_EMBEDDING_DIM, pad_idx=WORD_PAD_IDX)
    char_encoder = CharEncoder(vocab_size=CHAR_INPUT_DIM, embedding_dim=CHAR_EMBEDDING_DIM,
                               filter_size=CHAR_CNN_FILTER_SIZE, hidden_dim=WORD_EMBEDDING_DIM, pad_idx=CHAR_PAD_IDX)

    model = POSTagger(char_encoder=char_encoder, word_encoder=word_encoder, output_dim=OUTPUT_DIM,
                      hidden_dim=HIDDEN_DIM,
                      n_layers=N_LAYERS, dropout=DROPOUT)
    model.load_state_dict(model_state_dict)
    model.to(device)

    ans = ''
    for line in lines:
        words_with_UNK = line.strip().split(' ')
        max_word_len = max([len(word) for word in words_with_UNK])
        words = [word_dict[word] if word in word_dict else word_dict['<unk>'] for word in words_with_UNK]
        chars = []

        for word in words_with_UNK:
            chars_in_word = [char_dict[char] for char in word]
            padding_len = max_word_len - len(chars_in_word)
            for _ in range(padding_len):
                chars_in_word.append(CHAR_PAD_IDX)
            chars.append(chars_in_word)

        words_tensor = torch.tensor(words, dtype=torch.long).view(-1, 1).to(device)
        chars_tensor = torch.tensor(chars, dtype=torch.long).view(1, -1, max_word_len).to(device)

        with torch.no_grad():
            predictions = model(words_tensor, chars_tensor)
            predictions = predictions.view(-1, predictions.shape[-1]).cpu()

        assigned_tag_ix = np.argmax(predictions, axis=1).numpy()
        tagged_words = ['%s/%s ' % (words_with_UNK[i], ix_to_tag[assigned_tag_ix[i]]) for i in range(len(assigned_tag_ix))]
        for tagged_word in tagged_words:
            ans += tagged_word
        ans += '\n'

    with open(out_file, 'w') as f:
        f.write(ans)
        f.close()

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
