# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import sys
import torch
import torch.nn as nn
import torch.optim as optim

import datetime

start_time = datetime.datetime.now()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WordEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size, padding_idx=pad_idx)

    def forward(self, words):
        return self.embedding(words)


class CharEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, hidden_dim, pad_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size, padding_idx=pad_idx)
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, filter_size,
                              padding=(filter_size - 1) // 2)

    def forward(self, chars):
        batch_size, sent_len, word_len = chars.shape[0], chars.shape[1], chars.shape[2]

        embedded = self.embedding(chars) \
            .view(-1, word_len, self.embedding_dim) \
            .permute(0, 2, 1)
        embedded = self.conv(embedded) \
            .view(batch_size, sent_len, self.hidden_dim, word_len)
        embedded, _ = torch.max(embedded, dim=-1)
        return embedded.permute(1, 0, 2)


class POSTagger(nn.Module):
    def __init__(self, word_encoder, char_encoder, hidden_dim, output_dim,
                 n_layers, dropout):
        super().__init__()
        self.word_encoder = word_encoder
        self.char_encoder = char_encoder
        # assert word_encoder.embedding_dim == char_encoder.hidden_dim
        self.lstm = nn.LSTM(num_layers=n_layers,
                            input_size=2 * word_encoder.embedding_dim,
                            bidirectional=True,
                            hidden_size=hidden_dim)
        self.fc = nn.Linear(out_features=output_dim, in_features=hidden_dim * 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, words, chars):
        words_embedded, chars_embedded = self.dropout(self.word_encoder(words)), self.dropout(self.char_encoder(chars))

        word_rep = torch.cat((chars_embedded, words_embedded), dim=-1)
        hidden_state, _ = self.lstm(word_rep)

        return self.fc(self.dropout(hidden_state))


from torch.utils.data import Dataset, DataLoader


class POSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)



def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0.0

    for batch in iterator:
        words, chars, tags = batch['words'], batch['chars'], batch['tags'].contiguous().view(-1)

        optimizer.zero_grad()
        predictions = model(words, chars)
        pred_view_shape = predictions.shape[-1]
        predictions = predictions.view(-1, pred_view_shape)

        loss = criterion(predictions, tags)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0.0

    for batch in iterator:
        with torch.no_grad():
            words, chars, tags = batch['words'], batch['chars'], batch['tags'].contiguous().view(-1)

            predictions = model(words, chars)
            pred_view_shape = predictions.shape[-1]
            predictions = predictions.view(-1, pred_view_shape)

            loss = criterion(predictions, tags)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


word_dict, char_dict, tag_dict = {}, {}, {}
word_pad_ix, char_pad_ix, tag_pad_ix = 0, 0, 0


def clean_up(training_data, word_frequency):
    word_to_ix = {'<unk>': 0}
    char_to_ix = {}
    tag_to_ix = {}

    cleaned_training_data = []
    for sent in training_data:
        words = ['<unk>' if word_frequency[word] == 1 else word for word in sent['words']]
        for char_word in sent['chars']:
            for char in char_word:
                if char in char_to_ix:
                    continue
                else:
                    char_to_ix[char] = len(char_to_ix)
        for word in sent['words']:
            if word_frequency[word] != 1 and word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in sent['tags']:
            if tag in tag_to_ix:
                continue
            else:
                tag_to_ix[tag] = len(tag_to_ix)
        cleaned_training_data.append({'words': words, 'chars': sent['chars'], 'tags': sent['tags']})

    return cleaned_training_data, word_to_ix, char_to_ix, tag_to_ix


def collate_fn(batch):
    global word_dict, char_dict, tag_dict, word_pad_ix, char_pad_ix, tag_pad_ix, device
    max_word = 0
    max_char = 0
    words_list = []
    chars_list = []
    tags_list = []
    for sent in batch:
        words, chars = sent['words'], sent['chars']
        max_word = len(words) if len(words) > max_word else max_word
        for char_word in chars:
            max_char = len(char_word) if len(char_word) > max_char else max_char

    chars_padding = [char_pad_ix for i in range(max_char)]
    for sent in batch:
        words, chars, tags = sent['words'], sent['chars'], sent['tags']

        words_ix = [word_dict[word] for word in words]
        while len(words_ix) < max_word:
            words_ix.append(word_pad_ix)
        words_list.append(words_ix)

        tags_ix = [tag_dict[tag] for tag in tags]
        while len(tags_ix) < max_word:
            tags_ix.append(tag_pad_ix)
        tags_list.append(tags_ix)

        chars_ix = []
        for char_word in chars:
            char_word_ix = [char_dict[char] for char in char_word]
            if max_char > len(char_word_ix):
                for _ in range(max_char - len(char_word_ix)):
                    char_word_ix.append(char_pad_ix)
            chars_ix.append(char_word_ix)
        if max_word > len(chars_ix):
            for _ in range(max_word - len(chars_ix)):
                chars_ix.append(chars_padding)
        chars_list.append(chars_ix)

    words_tensor = torch.LongTensor(words_list).to(device).permute(1, 0)
    chars_tensor = torch.LongTensor(chars_list).to(device)
    tags_tensor = torch.LongTensor(tags_list).to(device).permute(1, 0)

    return {'words': words_tensor, 'chars': chars_tensor, 'tags': tags_tensor}


def train_model(train_file, model_file):
    f = open(train_file)
    lines = f.readlines()
    f.close()

    training_data = []
    word_frequency = {}
    for line in lines:
        words, tags, chars = [], [], []
        line = line.strip().split(' ')
        for token in line:
            word, _, tag = token.rpartition('/')
            words.append(word)
            tags.append(tag)
            chars.append([char for char in word])
            if word in word_frequency:
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1
        training_data.append({'words': words, 'chars': chars, 'tags': tags})

    global word_dict, char_dict, tag_dict
    data, word_dict, char_dict, tag_dict = clean_up(training_data, word_frequency)
    evaluating_data = sorted(data[:int(len(data) * 0.1)], key=lambda x: len(x['words']) * -1)
    training_data = sorted(data[int(len(data) * 0.1):], key=lambda x: len(x['words']) * -1)

    evaluating_iterator = DataLoader(POSDataset(data=evaluating_data), batch_size=64, collate_fn=collate_fn)
    train_iterator = DataLoader(POSDataset(data=training_data), batch_size=64, collate_fn=collate_fn)

    # Hyperparameters
    WORD_EMBEDDING_DIM = 100
    CHAR_EMBEDDING_DIM = 50
    CHAR_CNN_FILTER_SIZE = 3
    HIDDEN_DIM = 128
    N_LAYERS = 2
    DROPOUT = 0.25

    # Computed values
    WORD_INPUT_DIM = len(word_dict) + 1
    WORD_PAD_IDX = len(word_dict)
    CHAR_INPUT_DIM = len(char_dict) + 1
    CHAR_PAD_IDX = len(char_dict)
    OUTPUT_DIM = len(tag_dict) + 1
    TAG_PAD_IDX = len(tag_dict)

    global word_pad_ix, char_pad_ix, tag_pad_ix
    word_pad_ix, char_pad_ix, tag_pad_ix = WORD_PAD_IDX, CHAR_PAD_IDX, TAG_PAD_IDX

    word_encoder = WordEncoder(vocab_size=WORD_INPUT_DIM, embedding_dim=WORD_EMBEDDING_DIM, pad_idx=WORD_PAD_IDX)
    char_encoder = CharEncoder(vocab_size=CHAR_INPUT_DIM, embedding_dim=CHAR_EMBEDDING_DIM,
                               filter_size=CHAR_CNN_FILTER_SIZE, hidden_dim=WORD_EMBEDDING_DIM, pad_idx=CHAR_PAD_IDX)
    model = POSTagger(char_encoder=char_encoder, word_encoder=word_encoder, output_dim=OUTPUT_DIM,
                      hidden_dim=HIDDEN_DIM,
                      n_layers=N_LAYERS, dropout=DROPOUT).to(device)
    for _, param in model.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.1)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX).to(device)

    N_EPOCHS = 30 # changed
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, evaluating_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save((word_dict, tag_dict, char_dict, model.state_dict()), model_file)
        elif datetime.datetime.now() - start_time > datetime.timedelta(minutes=9, seconds=10):
            torch.save((word_dict, tag_dict, char_dict, model.state_dict()), model_file)
            return

        # print(epoch)
        # print(train_loss)
        # print(valid_loss)
        # print('----------------------------')

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
