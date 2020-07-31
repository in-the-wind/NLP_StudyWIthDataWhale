# Day4 基于深度学习的文本分类

​	与机器学习类似，在进行基于深度学习的文本分类时，我们同样可以进行特征提取+分类器的思路去处理这一问题。

​	对于深度学习的特征提取方式，一般有word2vec,doc2vec等等。前者是将一个词用一个vector表示，后者则是将整个文本用一个vector表示。在处理长文本时，可能后者的效果会更好。不过一般word2vec相对更加基础，也是一系列'tovec'方式的开山之作，因此，我们最开始都会首先接触到word2vector。

​	对于Word2vec，我们可以通过调用gensim中的word2vec包去训练。

```python
from gensim.models.word2vec import Word2Vec

num_features = 100     # Word vector dimensionality
num_workers = 8       # Number of threads to run in parallel

train_texts = list(map(lambda x: list(x.split()), train_texts))
model = Word2Vec(train_texts, workers=num_workers, size=num_features)
model.init_sims(replace=True)

# save model
model.save("./word2vec.bin")
```

训练完毕后，`model`可以以"`.bin`"文件存储。

这时，`model`实际上就相当于一个 key为 词，value为对应向量的字典。

自然地，一个句子，就变成了一个二维矩阵，我们可能就会想到用CNN去处理这个矩阵，也就是**TextCNN**.

# TextCNN

TextCNN利用CNN（卷积神经网络）进行文本特征抽取，不同大小的卷积核分别抽取n-gram特征，卷积计算出的特征图经过MaxPooling保留最大的特征值，然后将拼接成一个向量作为文本的表示。

这里我们基于TextCNN原始论文的设定，分别采用了100个大小为2,3,4的卷积核，最后得到的文本向量大小为100*3=300维。

这个代码我觉得确实是有点长的，看下来也感觉有点头疼。一个epoch大概用了2个小时的时间。


```python
import logging
import random

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set seed 
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# set cuda
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)
```

    2020-07-30 18:47:19,183 INFO: Use cuda: True, gpu id: 0.



```python
# split data to 10 fold
fold_num = 10
data_file = '../input/train_set.csv'
import pandas as pd


def all_data2fold(fold_num, num=200000):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]

    total = len(labels)

    index = list(range(total))
    np.random.shuffle(index)

    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)

        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

    return fold_data


fold_data = all_data2fold(10)
```

    2020-07-30 18:47:28,988 INFO: generated new fontManager
    2020-07-30 18:47:50,044 INFO: Fold lens [20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000]



```python
# build train, dev, test data
fold_id = 9

# dev
dev_data = fold_data[fold_id]

# train
train_texts = []
train_labels = []
for i in range(0, fold_id):
    data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])

train_data = {'label': train_labels, 'text': train_texts}

# test
test_data_file = '../input/test_a.csv'
f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
texts = f['text'].tolist()
test_data = {'label': [0] * len(texts), 'text': texts}
```


```python
# build vocab
from collections import Counter
from transformers import BasicTokenizer

basic_tokenizer = BasicTokenizer()


class Vocab():
    def __init__(self, train_data):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []

        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    def build_vocab(self, data):
        self.word_counter = Counter()

        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1

        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        self.label_counter = Counter(data['label'])

        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])

    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)


vocab = Vocab(train_data)
```

    2020-07-30 18:51:01,047 INFO: PyTorch version 1.3.1+cu100 available.
    2020-07-30 18:52:50,680 INFO: Build vocab: words 5978, labels 14.



```python
# build module
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden

        # compute attention
        outputs = torch.matmul(key, self.query)  # b x len

        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))

        attn_scores = F.softmax(masked_outputs, dim=1)  # b x len

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


# build word encoder
word2vec_path = '../input/word2vec.txt'
dropout = 0.15


class WordCNNEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordCNNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 100

        self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)

        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))

        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.filter_sizes = [2, 3, 4]  # n-gram window
        self.out_channel = 100
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True)
                                    for filter_size in self.filter_sizes])

    def forward(self, word_ids, extword_ids):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len
        # batch_masks: sen_num x sent_len
        sen_num, sent_len = word_ids.shape

        word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)

        batch_embed.unsqueeze_(1)  # sen_num x 1 x sent_len x 100

        pooled_outputs = []
        for i in range(len(self.filter_sizes)):
            filter_height = sent_len - self.filter_sizes[i] + 1
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)  # sen_num x out_channel x filter_height x 1

            mp = nn.MaxPool2d((filter_height, 1))  # (filter_height, filter_width)
            pooled = mp(hidden).reshape(sen_num,
                                        self.out_channel)  # sen_num x out_channel x 1 x 1 -> sen_num x out_channel

            pooled_outputs.append(pooled)

        reps = torch.cat(pooled_outputs, dim=1)  # sen_num x total_out_channel

        if self.training:
            reps = self.dropout(reps)

        return reps


# build sent encoder
sent_hidden_size = 256
sent_num_layers = 2


class SentEncoder(nn.Module):
    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens
```


```python
# build model
class Model(nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.sent_rep_size = 300
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordCNNEncoder(vocab)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))

        self.sent_encoder = SentEncoder(self.sent_rep_size)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        if use_cuda:
            self.to(device)

        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        logging.info('Build model with cnn word encoder, lstm sent encoder.')

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len

        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size

        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs


model = Model(vocab)
```

    2020-07-30 19:43:16,151 INFO: Load extword embed: words 5978, dims 100.
    2020-07-30 19:47:55,659 INFO: Build model with cnn word encoder, lstm sent encoder.
    2020-07-30 19:47:55,661 INFO: Model param num: 4.28 M.



```python
# build optimizer
learning_rate = 2e-4
decay = .75
decay_step = 1000


class Optimizer:
    def __init__(self, model_parameters):
        self.all_params = []
        self.optims = []
        self.schedulers = []

        for name, parameters in model_parameters.items():
            if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=learning_rate)
                self.optims.append(optim)

                l = lambda step: decay ** (step // decay_step)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                self.schedulers.append(scheduler)
                self.all_params.extend(parameters)

            else:
                Exception("no nameed parameters.")

        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res
```


```python
# build dataset
def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    words = text.strip().split()
    document_len = len(words)

    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        segments.append([len(segment), segment])

    assert len(segments) > 0
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments


def get_examples(data, vocab, max_sent_len=256, max_segment=8):
    label2id = vocab.label2id
    examples = []

    for text, label in zip(data['text'], data['label']):
        # label
        id = label2id(label)

        # words
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            word_ids = vocab.word2id(sent_words)
            extword_ids = vocab.extword2id(sent_words)
            doc.append([sent_len, word_ids, extword_ids])
        examples.append([id, len(doc), doc])

    logging.info('Total %d docs.' % len(examples))
    return examples
```


```python
# build loader

def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        docs = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield docs


def data_iter(data, batch_size, shuffle=True, noise=1.0):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle:
        np.random.shuffle(data)

    lengths = [example[1] for example in data]
    noisy_lengths = [- (l + np.random.uniform(- noise, noise)) for l in lengths]
    sorted_indices = np.argsort(noisy_lengths).tolist()
    sorted_data = [data[i] for i in sorted_indices]

    batched_data.extend(list(batch_slice(sorted_data, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)

    for batch in batched_data:
        yield batch
```


```python
# some function
from sklearn.metrics import f1_score, precision_score, recall_score


def get_score(y_ture, y_pred):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_ture, y_pred, average='macro') * 100
    p = precision_score(y_ture, y_pred, average='macro') * 100
    r = recall_score(y_ture, y_pred, average='macro') * 100

    return str((reformat(p, 2), reformat(r, 2), reformat(f1, 2))), reformat(f1, 2)


def reformat(num, n):
    return float(format(num, '0.' + str(n) + 'f'))
```


```python
# build trainer

import time
from sklearn.metrics import classification_report

clip = 5.0
epochs = 1
early_stops = 3
log_interval = 50

test_batch_size = 128
train_batch_size = 128

save_model = './cnn.bin'
save_test = './cnn.csv'

class Trainer():
    def __init__(self, model, vocab):
        self.model = model
        self.report = True

        self.train_data = get_examples(train_data, vocab)
        self.batch_num = int(np.ceil(len(self.train_data) / float(train_batch_size)))
        self.dev_data = get_examples(dev_data, vocab)

        # criterion
        self.criterion = nn.CrossEntropyLoss()

        # label name
        self.target_names = vocab.target_names

        # optimizer
        self.optimizer = Optimizer(model.all_parameters)

        # count
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.last_epoch = epochs

    def train(self):
        logging.info('Start training...')
        for epoch in range(1, epochs + 1):
            train_f1 = self._train(epoch)

            dev_f1 = self._eval(epoch)

            if self.best_dev_f1 <= dev_f1:
                logging.info(
                    "Exceed history dev = %.2f, current dev = %.2f" % (self.best_dev_f1, dev_f1))
                torch.save(self.model.state_dict(), save_model)

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == early_stops:
                    logging.info(
                        "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
                            epoch - early_stops, self.best_train_f1, self.best_dev_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(save_model))
        self._eval(self.last_epoch + 1, test=True)
    
    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, train_batch_size, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)
            batch_outputs = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=clip)
            for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                optimizer.step()
                scheduler.step()
            self.optimizer.zero_grad()

            self.step += 1

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time

                lrs = self.optimizer.get_lr()
                logging.info(
                    '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                        epoch, self.step, batch_idx, self.batch_num, lrs,
                        losses / log_interval,
                        elapsed / log_interval))

                losses = 0
                start_time = time.time()

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)

        logging.info(
            '| epoch {:3d} | score {} | f1 {} | loss {:.4f} | time {:.2f}'.format(epoch, score, f1,
                                                                                  overall_losses,
                                                                                  during_time))
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1

    def _eval(self, epoch, test=False):
        self.model.eval()
        start_time = time.time()

        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_data in data_iter(self.dev_data, test_batch_size, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

            score, f1 = get_score(y_true, y_pred)

            during_time = time.time() - start_time
            
            if test:
                df = pd.DataFrame({'label': y_pred})
                df.to_csv(save_test, index=False, sep=',')
            else:
                logging.info(
                    '| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'.format(epoch, score, f1,
                                                                              during_time))
                if set(y_true) == set(y_pred) and self.report:
                    report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                    logging.info('\n' + report)

        return f1

    def batch2tensor(self, batch_data):
        '''
            [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
        '''
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        max_doc_len = max(doc_lens)
        max_sent_len = max(doc_max_sent_len)

        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                sent_data = batch_data[b][2][sent_idx]
                for word_idx in range(sent_data[0]):
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_inputs2[b, sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[b, sent_idx, word_idx] = 1

        if use_cuda:
            batch_inputs1 = batch_inputs1.to(device)
            batch_inputs2 = batch_inputs2.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

        return (batch_inputs1, batch_inputs2, batch_masks), batch_labels
```


```python
# train
trainer = Trainer(model, vocab)
trainer.train()
```

    2020-07-30 20:09:44,286 INFO: Total 180000 docs.
    2020-07-30 20:12:08,861 INFO: Total 20000 docs.
    2020-07-30 20:12:08,862 INFO: Start training...
    2020-07-30 20:16:51,723 INFO: | epoch   1 | step  50 | batch  50/1407 | lr 0.00020 | loss 2.2958 | s/batch 5.66
    2020-07-30 20:21:01,752 INFO: | epoch   1 | step 100 | batch 100/1407 | lr 0.00020 | loss 1.8964 | s/batch 5.00
    2020-07-30 20:24:56,957 INFO: | epoch   1 | step 150 | batch 150/1407 | lr 0.00020 | loss 1.3986 | s/batch 4.70
    2020-07-30 20:28:37,152 INFO: | epoch   1 | step 200 | batch 200/1407 | lr 0.00020 | loss 1.1400 | s/batch 4.40
    2020-07-30 20:32:27,653 INFO: | epoch   1 | step 250 | batch 250/1407 | lr 0.00020 | loss 0.9660 | s/batch 4.61
    2020-07-30 20:35:52,549 INFO: | epoch   1 | step 300 | batch 300/1407 | lr 0.00020 | loss 0.8222 | s/batch 4.10
    2020-07-30 20:39:46,738 INFO: | epoch   1 | step 350 | batch 350/1407 | lr 0.00020 | loss 0.7642 | s/batch 4.68
    2020-07-30 20:44:01,642 INFO: | epoch   1 | step 400 | batch 400/1407 | lr 0.00020 | loss 0.6622 | s/batch 5.10
    2020-07-30 20:48:19,112 INFO: | epoch   1 | step 450 | batch 450/1407 | lr 0.00020 | loss 0.6331 | s/batch 5.15
    2020-07-30 20:52:36,358 INFO: | epoch   1 | step 500 | batch 500/1407 | lr 0.00020 | loss 0.5494 | s/batch 5.14
    2020-07-30 20:56:53,479 INFO: | epoch   1 | step 550 | batch 550/1407 | lr 0.00020 | loss 0.5747 | s/batch 5.14
    2020-07-30 21:01:16,823 INFO: | epoch   1 | step 600 | batch 600/1407 | lr 0.00020 | loss 0.5533 | s/batch 5.27
    2020-07-30 21:05:59,736 INFO: | epoch   1 | step 650 | batch 650/1407 | lr 0.00020 | loss 0.5194 | s/batch 5.66
    2020-07-30 21:09:35,062 INFO: | epoch   1 | step 700 | batch 700/1407 | lr 0.00020 | loss 0.4736 | s/batch 4.31
    2020-07-30 21:13:30,900 INFO: | epoch   1 | step 750 | batch 750/1407 | lr 0.00020 | loss 0.4723 | s/batch 4.72
    2020-07-30 21:17:13,890 INFO: | epoch   1 | step 800 | batch 800/1407 | lr 0.00020 | loss 0.4468 | s/batch 4.46
    2020-07-30 21:20:59,268 INFO: | epoch   1 | step 850 | batch 850/1407 | lr 0.00020 | loss 0.4195 | s/batch 4.51
    2020-07-30 21:24:57,133 INFO: | epoch   1 | step 900 | batch 900/1407 | lr 0.00020 | loss 0.4465 | s/batch 4.76
    2020-07-30 21:29:18,349 INFO: | epoch   1 | step 950 | batch 950/1407 | lr 0.00020 | loss 0.4427 | s/batch 5.22
    2020-07-30 21:32:54,144 INFO: | epoch   1 | step 1000 | batch 1000/1407 | lr 0.00015 | loss 0.3976 | s/batch 4.32
    2020-07-30 21:36:04,900 INFO: | epoch   1 | step 1050 | batch 1050/1407 | lr 0.00015 | loss 0.4148 | s/batch 3.82
    2020-07-30 21:40:05,877 INFO: | epoch   1 | step 1100 | batch 1100/1407 | lr 0.00015 | loss 0.4008 | s/batch 4.82
    2020-07-30 21:44:06,238 INFO: | epoch   1 | step 1150 | batch 1150/1407 | lr 0.00015 | loss 0.3620 | s/batch 4.81
    2020-07-30 21:48:05,447 INFO: | epoch   1 | step 1200 | batch 1200/1407 | lr 0.00015 | loss 0.3784 | s/batch 4.78
    2020-07-30 21:51:59,470 INFO: | epoch   1 | step 1250 | batch 1250/1407 | lr 0.00015 | loss 0.3918 | s/batch 4.68
    2020-07-30 21:56:14,560 INFO: | epoch   1 | step 1300 | batch 1300/1407 | lr 0.00015 | loss 0.4032 | s/batch 5.10
    2020-07-30 21:59:10,182 INFO: | epoch   1 | step 1350 | batch 1350/1407 | lr 0.00015 | loss 0.3621 | s/batch 3.51
    2020-07-30 22:02:51,309 INFO: | epoch   1 | step 1400 | batch 1400/1407 | lr 0.00015 | loss 0.3523 | s/batch 4.42
    2020-07-30 22:03:33,440 INFO: | epoch   1 | score (75.89, 63.21, 67.43) | f1 67.43 | loss 0.6747 | time 6684.32
    2020-07-30 22:03:33,868 INFO: 
                  precision    recall  f1-score   support
    
              科技     0.7419    0.8494    0.7921     35027
              股票     0.7935    0.8785    0.8338     33251
              体育     0.8726    0.9432    0.9065     28283
              娱乐     0.7894    0.8289    0.8086     19920
              时政     0.7522    0.7252    0.7384     13515
              社会     0.7334    0.6715    0.7011     11009
              教育     0.8566    0.7452    0.7970      8987
              财经     0.7293    0.4459    0.5534      7957
              家居     0.6988    0.5815    0.6348      7063
              游戏     0.8156    0.6020    0.6927      5291
              房产     0.8793    0.6825    0.7685      4428
              时尚     0.6454    0.3552    0.4582      2818
              彩票     0.8103    0.4029    0.5382      1633
              星座     0.5067    0.1381    0.2171       818
    
        accuracy                         0.7871    180000
       macro avg     0.7589    0.6321    0.6743    180000
    weighted avg     0.7845    0.7871    0.7799    180000
    
    2020-07-30 22:15:38,682 INFO: | epoch   1 | dev | score (89.75, 84.77, 85.97) | f1 85.97 | time 724.81
    2020-07-30 22:15:38,736 INFO: 
                  precision    recall  f1-score   support
    
              科技     0.9503    0.8800    0.9138      3891
              股票     0.9230    0.9407    0.9318      3694
              体育     0.9790    0.9812    0.9801      3142
              娱乐     0.9098    0.9521    0.9304      2213
              时政     0.7942    0.9254    0.8548      1501
              社会     0.8179    0.8520    0.8346      1223
              教育     0.9340    0.9078    0.9207       998
              财经     0.8893    0.7455    0.8111       884
              家居     0.8426    0.8878    0.8646       784
              游戏     0.8639    0.8978    0.8805       587
              房产     0.9165    0.9593    0.9374       492
              时尚     0.8732    0.7923    0.8308       313
              彩票     0.9408    0.8457    0.8908       188
              星座     0.9310    0.3000    0.4538        90
    
        accuracy                         0.9107     20000
       macro avg     0.8975    0.8477    0.8597     20000
    weighted avg     0.9130    0.9107    0.9100     20000
    
    2020-07-30 22:15:38,737 INFO: Exceed history dev = 0.00, current dev = 85.97



```python
# test
trainer.test()
```


```python
cnn_data=pd.read_csv(save_test,sep=',')
```


```python
print(cnn_data.describe())
```

                  label
    count  20000.000000
    mean       3.209950
    std        2.997919
    min        0.000000
    25%        1.000000
    50%        2.000000
    75%        5.000000
    max       13.000000



```python
model=torch.load(save_model)
```


```python
print(model.values()
```

    odict_values([tensor([[ 3.2521e-03,  7.5364e-04, -4.7066e-03,  ...,  3.7055e-03,
              2.0840e-03,  4.2703e-03],
            [ 1.0137e+00, -1.6919e+00, -1.1315e+00,  ...,  9.3965e-01,
             -2.6428e-02,  9.5890e-01],
            [-1.1926e+00, -4.4183e-01, -9.6629e-01,  ...,  2.5037e-01,
             -1.3964e-01, -5.4609e-01],
            ...,
            [ 1.2816e-01,  2.0311e-01, -1.3755e+00,  ...,  7.3771e-02,
             -2.4486e-01, -1.1128e+00],
            [-6.9021e-01,  1.1232e+00,  9.0740e-01,  ..., -3.3285e-01,
             -1.7871e-02,  1.1273e+00],
            [ 1.3178e-02, -1.8755e+00, -9.3433e-02,  ...,  1.9075e-01,
             -4.2521e-01, -2.5208e+00]], device='cuda:0'), tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
              0.0000e+00,  0.0000e+00],
            [ 1.2652e-01,  3.5368e-04,  1.4295e-01,  ..., -2.3008e-01,
             -2.5458e-01, -1.4965e-01],
            [-2.5648e+00, -8.2837e-01,  5.1653e-01,  ..., -1.0160e+00,
              1.2390e+00, -1.8944e+00],
            ...,
            [ 5.4010e-01, -2.9805e-01,  1.3584e+00,  ...,  1.2431e+00,
             -2.8292e-01,  1.2827e+00],
            [-5.7362e-01,  3.1667e-01,  7.4657e-01,  ..., -6.7222e-01,
             -5.9554e-03, -6.2650e-01],
            [ 2.2282e-01,  5.2851e-01, -2.6891e-01,  ..., -6.5441e-01,
             -2.0342e+00, -1.2090e-01]], device='cuda:0'), tensor([[[[-0.0916,  0.0255,  0.0733,  ..., -0.0772, -0.0774, -0.0236],
              [-0.0143, -0.0454, -0.0121,  ..., -0.0670, -0.0582, -0.0473]]],


​    

            [[[ 0.0207, -0.0066, -0.0903,  ..., -0.0506,  0.0885,  0.0439],
              [-0.0509,  0.0559, -0.0523,  ..., -0.0013, -0.0313, -0.0013]]],


​    

            [[[-0.0533, -0.0272, -0.0303,  ..., -0.0006, -0.0313, -0.0708],
              [ 0.0211, -0.0695,  0.0142,  ..., -0.0431,  0.0035, -0.0288]]],


​    

            ...,


​    

            [[[ 0.0120,  0.0599, -0.0267,  ...,  0.0570, -0.0375, -0.0044],
              [-0.0419, -0.0401, -0.0167,  ..., -0.0102,  0.0404,  0.0430]]],


​    

            [[[-0.0505,  0.0609, -0.0105,  ...,  0.0068,  0.0684, -0.0542],
              [ 0.0038,  0.0505, -0.0329,  ...,  0.0321, -0.0153, -0.0219]]],


​    

            [[[ 0.0553,  0.0622, -0.0060,  ..., -0.0119, -0.0699,  0.0470],
              [-0.0411,  0.0518,  0.0238,  ..., -0.0333, -0.0812,  0.0316]]]],
           device='cuda:0'), tensor([-7.4347e-02,  3.8265e-02, -5.4233e-02, -6.4228e-02,  1.8969e-02,
            -5.0762e-02,  7.4136e-03, -3.8194e-03, -7.0754e-02, -3.4902e-02,
             1.8701e-02, -2.2900e-02, -9.6061e-02, -5.5352e-02,  1.3461e-03,
             2.9631e-02,  2.3124e-02, -2.0788e-02, -3.0463e-02, -1.8794e-02,
            -4.0115e-02, -6.6532e-02, -4.9336e-02, -7.7866e-03, -5.6662e-03,
             5.5120e-02, -6.2003e-02, -2.8314e-02,  2.9712e-02, -3.1345e-02,
             3.4537e-02,  5.8286e-02,  2.9716e-02, -7.3982e-02, -3.1525e-02,
             3.8163e-02, -1.1177e-02,  1.8200e-02, -6.1242e-02, -7.4159e-02,
            -5.9231e-02, -7.4723e-02, -4.5513e-02,  1.0624e-02,  4.2051e-02,
             3.3739e-02, -2.6701e-02,  2.9022e-02,  3.9111e-02, -8.7633e-02,
            -7.3828e-02, -6.5197e-02,  4.6095e-02,  5.2772e-03,  5.2979e-02,
             3.7697e-02, -5.0607e-02, -2.3304e-02, -1.6956e-02, -9.1008e-02,
             3.7772e-02, -7.4730e-02, -8.0711e-02,  3.4440e-02,  5.6269e-02,
            -7.9516e-02,  4.8238e-02,  3.0859e-03,  3.8304e-02,  4.1571e-02,
            -3.0516e-02, -7.2469e-03, -5.6932e-02, -5.2092e-02,  2.3202e-02,
            -2.3733e-02, -2.2535e-02, -6.8650e-02, -6.9479e-02, -2.7590e-02,
             6.3014e-02, -6.0900e-02, -1.5440e-02, -5.3133e-02,  4.1281e-02,
             1.2976e-05, -7.8898e-02, -6.6217e-02, -5.9942e-02, -7.6352e-02,
             5.7800e-02,  4.1127e-02, -2.6031e-04,  2.3211e-02, -7.8055e-02,
             3.7924e-02,  2.2102e-03, -2.0042e-02,  3.8456e-02, -3.9234e-02],
           device='cuda:0'), tensor([[[[-0.0467, -0.0541,  0.0371,  ..., -0.0430, -0.0030,  0.0178],
              [ 0.0274, -0.0332, -0.0108,  ...,  0.0041,  0.0118,  0.0064],
              [ 0.0349, -0.0194,  0.0382,  ..., -0.0188,  0.0374,  0.0137]]],


​    

            [[[ 0.0391, -0.0024, -0.0016,  ...,  0.0064,  0.0158, -0.0386],
              [-0.0217, -0.0106,  0.0305,  ..., -0.0169, -0.0449,  0.0178],
              [ 0.0136, -0.0481, -0.0270,  ...,  0.0349,  0.0333, -0.0423]]],


​    

            [[[-0.0611, -0.0098,  0.0461,  ..., -0.0146,  0.0016,  0.0424],
              [-0.0437,  0.0559,  0.0482,  ..., -0.0023, -0.0262, -0.0091],
              [-0.0031,  0.0608,  0.0644,  ..., -0.0643, -0.0446,  0.0418]]],


​    

            ...,


​    

            [[[ 0.0456,  0.0167,  0.0399,  ..., -0.0068, -0.0280, -0.0608],
              [ 0.0156, -0.0067, -0.0186,  ..., -0.0407,  0.0401, -0.0297],
              [-0.0586, -0.0408,  0.0253,  ..., -0.0722,  0.0222, -0.0529]]],


​    

            [[[ 0.0196,  0.0158, -0.0131,  ...,  0.0179, -0.0176,  0.0100],
              [ 0.0149, -0.0106,  0.0296,  ...,  0.0177, -0.0090,  0.0278],
              [-0.0071, -0.0701, -0.0163,  ..., -0.0409, -0.0094, -0.0047]]],


​    

            [[[-0.0499,  0.0353, -0.0039,  ..., -0.0007, -0.0404,  0.0262],
              [-0.0409,  0.0224, -0.0345,  ..., -0.0082, -0.0403,  0.0345],
              [ 0.0536,  0.0407, -0.0554,  ...,  0.0350,  0.0429,  0.0029]]]],
           device='cuda:0'), tensor([ 0.0113,  0.0390, -0.0424,  0.0018,  0.0131,  0.0357,  0.0095,  0.0390,
             0.0049, -0.0079, -0.0413, -0.0076,  0.0116, -0.0467,  0.0411,  0.0453,
            -0.0543,  0.0180,  0.0046,  0.0175, -0.0586, -0.0390,  0.0090, -0.0194,
             0.0376,  0.0369, -0.0420, -0.0385, -0.0425, -0.0141,  0.0247, -0.0047,
            -0.0348,  0.0215, -0.0092,  0.0176,  0.0036,  0.0143,  0.0276, -0.0533,
            -0.0124,  0.0373,  0.0304,  0.0038,  0.0117, -0.0360, -0.0549, -0.0044,
            -0.0713, -0.0634,  0.0278,  0.0128, -0.0407,  0.0051,  0.0394, -0.0285,
             0.0202,  0.0308, -0.0216, -0.0642,  0.0286,  0.0018,  0.0132, -0.0269,
            -0.0498, -0.0402, -0.0777,  0.0185, -0.0381,  0.0056, -0.0481, -0.0629,
             0.0194,  0.0085, -0.0330, -0.0180,  0.0276,  0.0127, -0.0367, -0.0444,
            -0.0573, -0.0404, -0.0607, -0.0358, -0.0589,  0.0379, -0.0539,  0.0143,
            -0.0599, -0.0243,  0.0296, -0.0485,  0.0447, -0.0176, -0.0677, -0.0341,
            -0.0728,  0.0162, -0.0692, -0.0127], device='cuda:0'), tensor([[[[-2.0529e-02, -5.3050e-03,  4.2921e-02,  ...,  3.9122e-02,
                3.6441e-02,  8.3580e-03],
              [ 9.8869e-03,  2.6355e-02, -9.5724e-03,  ...,  2.8477e-02,
               -2.2936e-02, -2.6076e-02],
              [ 2.7103e-02, -1.7606e-02,  2.6477e-02,  ...,  3.8809e-02,
               -2.2190e-02, -5.7815e-02],
              [ 3.9961e-02,  1.3962e-02, -3.3627e-02,  ..., -2.3707e-02,
               -2.5524e-02, -6.7391e-03]]],


​    

            [[[ 4.8128e-02, -5.5604e-02, -1.0406e-02,  ..., -3.6369e-02,
                3.2669e-02, -1.0368e-02],
              [ 5.6223e-02,  1.9224e-02,  1.0819e-02,  ...,  2.6031e-02,
               -5.2118e-02,  1.5880e-02],
              [ 7.4044e-02, -4.6061e-02, -2.6964e-02,  ..., -1.4186e-02,
               -4.5526e-02, -2.2541e-02],
              [ 2.8745e-02, -3.8164e-02, -1.6185e-02,  ...,  3.3272e-03,
                1.6386e-02,  7.3494e-03]]],


​    

            [[[-2.0962e-02,  1.5237e-03,  4.0812e-02,  ..., -3.1163e-02,
                2.5531e-02, -2.8056e-04],
              [-1.9870e-02, -2.6083e-02,  1.8152e-02,  ..., -8.9609e-03,
               -1.8797e-02, -8.7635e-03],
              [-8.8484e-03,  4.4094e-02, -1.9931e-03,  ..., -2.6260e-02,
               -4.2407e-02, -1.4559e-02],
              [-1.7141e-02,  4.0049e-02, -1.6849e-02,  ...,  9.1136e-03,
               -6.0265e-05, -1.0892e-02]]],


​    

            ...,


​    

            [[[-3.5417e-02,  3.5427e-03, -4.2035e-02,  ...,  3.0772e-02,
               -1.5499e-02, -2.0961e-02],
              [ 4.8999e-03, -6.4739e-02, -3.2256e-02,  ..., -1.6171e-02,
                5.2081e-03, -2.7888e-02],
              [-1.0474e-02,  1.7781e-02, -2.7949e-02,  ...,  3.2330e-02,
               -3.8836e-03, -8.5368e-03],
              [-9.1088e-03, -3.6806e-02, -2.4736e-03,  ...,  2.0461e-02,
               -4.5410e-02, -3.1112e-02]]],


​    

            [[[ 1.7497e-02, -2.5088e-02,  1.0955e-02,  ..., -5.1669e-03,
                4.3349e-03,  3.1592e-03],
              [ 3.9813e-02, -2.4130e-02,  1.2550e-02,  ...,  2.1945e-02,
                4.1156e-02,  8.9819e-03],
              [-7.0571e-03, -5.9334e-02, -1.0302e-02,  ..., -8.2936e-03,
                2.5566e-02,  4.3284e-02],
              [-4.1985e-02, -6.1314e-02, -5.9746e-03,  ...,  4.9788e-02,
                1.7613e-02, -1.3089e-02]]],


​    

            [[[-1.8432e-03,  8.1780e-03,  1.8403e-02,  ...,  5.4306e-02,
                1.7489e-04,  7.5156e-04],
              [ 1.4763e-02,  3.8983e-02, -7.1858e-04,  ..., -5.3340e-03,
               -1.0115e-02, -1.0268e-02],
              [-2.2521e-02,  4.8917e-02, -6.9425e-03,  ...,  3.1184e-02,
               -4.3385e-02,  5.2406e-03],
              [-5.0969e-02,  2.3552e-02,  2.1852e-02,  ...,  2.5215e-02,
                3.8510e-02,  9.2751e-04]]]], device='cuda:0'), tensor([-0.0025,  0.0304,  0.0101, -0.0239,  0.0163, -0.0153, -0.0250, -0.0474,
            -0.0369, -0.0027, -0.0194,  0.0333, -0.0393, -0.0495, -0.0347, -0.0391,
             0.0198,  0.0202, -0.0688,  0.0126,  0.0017, -0.0142, -0.0179, -0.0206,
             0.0356, -0.0418, -0.0628, -0.0249,  0.0322, -0.0113, -0.0092, -0.0498,
            -0.0269, -0.0166,  0.0242,  0.0079,  0.0274, -0.0286, -0.0406, -0.0258,
             0.0247,  0.0317,  0.0165, -0.0307, -0.0635, -0.0008, -0.0048,  0.0189,
             0.0091, -0.0482, -0.0561, -0.0441,  0.0013, -0.0483,  0.0141,  0.0277,
             0.0299, -0.0039,  0.0065, -0.0378, -0.0306, -0.0221, -0.0383, -0.0451,
             0.0024, -0.0541,  0.0302, -0.0139,  0.0243, -0.0096,  0.0347, -0.0578,
             0.0336,  0.0331, -0.0060,  0.0107,  0.0071, -0.0192, -0.0174,  0.0048,
             0.0209, -0.0011, -0.0434,  0.0074,  0.0023,  0.0192, -0.0396,  0.0044,
            -0.0512, -0.0469,  0.0152, -0.0218,  0.0178, -0.0443,  0.0091, -0.0449,
             0.0112, -0.0295, -0.0416, -0.0476], device='cuda:0'), tensor([[-0.0403, -0.0549, -0.0241,  ...,  0.0676,  0.0322,  0.0579],
            [ 0.0469,  0.0446, -0.0267,  ...,  0.0060, -0.0424, -0.0285],
            [-0.0243, -0.0291,  0.0571,  ...,  0.0360, -0.0031,  0.0470],
            ...,
            [ 0.0082,  0.0597,  0.0208,  ..., -0.0410,  0.0302, -0.0103],
            [ 0.0584,  0.0419,  0.0315,  ..., -0.0330,  0.0182, -0.0079],
            [ 0.0246, -0.0517, -0.0123,  ..., -0.0564,  0.0685,  0.0273]],
           device='cuda:0'), tensor([[ 0.0531, -0.0727,  0.0004,  ..., -0.0461,  0.0243, -0.0370],
            [-0.0400, -0.0159, -0.0051,  ...,  0.0456,  0.0410, -0.0273],
            [-0.0089, -0.0415, -0.0216,  ...,  0.0132,  0.0582,  0.0066],
            ...,
            [ 0.0579,  0.0382,  0.0366,  ...,  0.0189, -0.0599, -0.0452],
            [-0.0338,  0.0291,  0.0480,  ..., -0.0512, -0.0552,  0.0535],
            [-0.0487,  0.0511,  0.0429,  ..., -0.0125, -0.0134, -0.0192]],
           device='cuda:0'), tensor([-0.0094,  0.0041, -0.0463,  ..., -0.0035, -0.0213,  0.0219],
           device='cuda:0'), tensor([-0.0032, -0.0071,  0.0402,  ..., -0.0279,  0.0302, -0.0416],
           device='cuda:0'), tensor([[ 0.0346, -0.0612,  0.0204,  ...,  0.0370,  0.0114, -0.0432],
            [-0.0547,  0.0245, -0.0164,  ...,  0.0566, -0.0129, -0.0244],
            [ 0.0506, -0.0238,  0.0671,  ..., -0.0442,  0.0211, -0.0316],
            ...,
            [ 0.0163, -0.0250,  0.0245,  ...,  0.0509,  0.0065, -0.0569],
            [-0.0499, -0.0360,  0.0249,  ...,  0.0384,  0.0586,  0.0370],
            [ 0.0472, -0.0422,  0.0425,  ...,  0.0074, -0.0407,  0.0393]],
           device='cuda:0'), tensor([[ 0.0244, -0.0059, -0.0409,  ...,  0.0177, -0.0655,  0.0464],
            [-0.0456,  0.0102,  0.0390,  ...,  0.0192, -0.0503, -0.0433],
            [-0.0511, -0.0421, -0.0107,  ...,  0.0215, -0.0211,  0.0143],
            ...,
            [-0.0390,  0.0103,  0.0353,  ..., -0.0464, -0.0372, -0.0347],
            [-0.0654, -0.0287, -0.0519,  ...,  0.0486, -0.0597, -0.0539],
            [-0.0210,  0.0025,  0.0376,  ..., -0.0146, -0.0321, -0.0003]],
           device='cuda:0'), tensor([-0.0063, -0.0470, -0.0463,  ...,  0.0477,  0.0194,  0.0476],
           device='cuda:0'), tensor([ 0.0063,  0.0080,  0.0537,  ..., -0.0289, -0.0599,  0.0350],
           device='cuda:0'), tensor([[ 0.0077, -0.0673,  0.0317,  ..., -0.0365,  0.0490,  0.0595],
            [ 0.0123, -0.0955,  0.0097,  ...,  0.0882, -0.0463, -0.0779],
            [ 0.0494, -0.0128,  0.0115,  ..., -0.0250,  0.0415, -0.0356],
            ...,
            [-0.0098, -0.0462, -0.0334,  ...,  0.0763,  0.0343,  0.0140],
            [ 0.0031, -0.0086, -0.0515,  ..., -0.0187, -0.0249, -0.0231],
            [-0.0462, -0.0253, -0.0616,  ...,  0.0356, -0.0092, -0.0427]],
           device='cuda:0'), tensor([[-0.0069,  0.0631,  0.0060,  ..., -0.0254, -0.0242, -0.0036],
            [-0.0368, -0.0275,  0.0008,  ...,  0.0428,  0.0237,  0.0382],
            [-0.0357, -0.0375,  0.0332,  ...,  0.0295, -0.0094,  0.0638],
            ...,
            [ 0.0482,  0.0084, -0.0279,  ..., -0.0141, -0.0147, -0.0464],
            [ 0.0290, -0.0590, -0.0336,  ..., -0.0065,  0.0289, -0.0477],
            [-0.0298,  0.0249, -0.0368,  ...,  0.0265,  0.0383, -0.0499]],
           device='cuda:0'), tensor([ 0.0667,  0.0706, -0.0399,  ...,  0.0518, -0.0023, -0.0266],
           device='cuda:0'), tensor([ 0.0106,  0.0385, -0.0551,  ..., -0.0046,  0.0525,  0.0383],
           device='cuda:0'), tensor([[ 0.0088,  0.0311, -0.0244,  ...,  0.0020,  0.0146,  0.0395],
            [-0.0224,  0.0245, -0.0445,  ...,  0.0132, -0.0608, -0.0834],
            [-0.0679, -0.0729, -0.0699,  ..., -0.0203, -0.0229,  0.0523],
            ...,
            [-0.0241, -0.0414,  0.0036,  ..., -0.0205,  0.0757, -0.0498],
            [ 0.0084, -0.0416, -0.0574,  ..., -0.0300, -0.0387, -0.0531],
            [ 0.0157,  0.0378,  0.0032,  ...,  0.0351,  0.0261, -0.0403]],
           device='cuda:0'), tensor([[ 0.0221, -0.0528, -0.0329,  ..., -0.0138,  0.0611,  0.0159],
            [ 0.0451,  0.0236,  0.0509,  ..., -0.0289,  0.0210,  0.0203],
            [ 0.0354,  0.0269,  0.0257,  ...,  0.0274, -0.0454, -0.0174],
            ...,
            [ 0.0319,  0.0278, -0.0286,  ...,  0.0408,  0.0122,  0.0023],
            [ 0.0552,  0.0053, -0.0622,  ...,  0.0375, -0.0516, -0.0168],
            [-0.0413,  0.0462,  0.0450,  ..., -0.0031,  0.0027,  0.0239]],
           device='cuda:0'), tensor([ 0.0465, -0.0250, -0.0282,  ..., -0.0105, -0.0261, -0.0260],
           device='cuda:0'), tensor([ 0.0445,  0.0594, -0.0102,  ...,  0.0543, -0.0436,  0.0603],
           device='cuda:0'), tensor([[-2.9397e-02, -9.3624e-02, -6.0017e-02,  ...,  4.5834e-02,
              5.1474e-02, -4.4038e-02],
            [-3.5850e-02, -1.2897e-02, -2.0828e-02,  ..., -2.6666e-02,
             -2.7864e-05,  3.3211e-02],
            [ 3.7721e-02,  2.0504e-02, -8.3642e-03,  ...,  3.8016e-02,
              2.2149e-02, -1.0187e-02],
            ...,
            [ 3.7921e-02, -2.8563e-02, -7.8993e-02,  ..., -1.3236e-02,
             -2.5834e-02, -1.9589e-02],
            [-3.1271e-03, -1.1774e-01,  5.1372e-03,  ..., -2.1624e-02,
              3.9849e-02, -2.1825e-02],
            [-4.9650e-02,  2.2215e-02, -7.3707e-03,  ..., -2.5669e-02,
             -9.2267e-02, -8.8966e-02]], device='cuda:0'), tensor([-3.7620e-03,  3.0529e-05, -4.7795e-03, -4.9100e-03,  5.6958e-04,
             2.7825e-03,  6.8437e-03, -1.9781e-03,  5.7933e-04,  4.1568e-03,
            -4.3000e-05, -5.3404e-03, -7.5391e-04,  4.5917e-04, -1.1816e-03,
            -5.1019e-03,  6.8433e-03,  3.7227e-03, -1.5370e-03,  8.3895e-04,
             4.3436e-03,  6.5808e-03, -8.3308e-03, -5.1251e-03, -1.0122e-03,
             5.1275e-03,  1.5730e-03, -6.4007e-03,  1.7869e-03,  1.2319e-03,
             8.5859e-04,  3.3636e-04,  4.1223e-03,  1.2269e-03,  3.0934e-03,
            -2.4814e-03, -2.6050e-03, -1.7688e-03, -8.2464e-04, -1.9650e-03,
            -2.1586e-03,  5.8287e-04,  3.3789e-03, -7.7421e-04, -3.8647e-03,
             8.7981e-04, -2.6988e-03, -1.7255e-03,  3.1634e-03, -3.5904e-03,
             5.4410e-03,  3.6812e-04, -3.9470e-03, -1.6652e-03,  6.3514e-04,
            -2.2998e-03,  4.7145e-03,  2.6980e-03, -8.0077e-03, -2.6059e-03,
            -4.0590e-03, -6.5690e-04, -4.7234e-04, -3.2376e-03, -6.4456e-03,
            -3.4936e-03, -2.1344e-03,  3.3728e-03,  1.2436e-03,  2.5216e-03,
             1.7100e-03, -4.4365e-03,  6.5973e-04,  6.9207e-03, -3.7797e-03,
             1.7480e-03,  6.9004e-04, -5.2716e-03, -6.4200e-03,  1.6490e-03,
             8.5452e-05, -5.7125e-04,  4.6653e-03, -1.3673e-03,  8.3027e-03,
            -3.0970e-03,  7.9048e-03,  5.4762e-03, -5.6352e-03,  8.1782e-03,
             3.8276e-03, -1.4493e-03, -1.3635e-03, -1.6467e-03,  5.9721e-03,
             2.9245e-03,  7.8506e-03,  6.5286e-03, -6.3654e-03,  8.6880e-04,
            -3.8583e-03, -8.3319e-03,  5.0239e-03,  3.8005e-03, -5.0847e-03,
             3.6519e-03, -7.3355e-03,  1.5772e-03,  7.8247e-03, -2.1358e-03,
            -2.7636e-03,  1.3767e-03,  6.7010e-04,  4.9042e-03,  7.8149e-03,
             1.1010e-03, -1.9786e-03,  7.8379e-03,  3.9980e-04, -4.7061e-04,
            -2.7243e-03,  1.6939e-03, -6.4186e-03,  1.3937e-03, -6.6315e-03,
             4.7637e-03,  3.5519e-03, -4.6236e-04,  2.7775e-03, -1.8221e-03,
            -6.8957e-04,  3.4511e-03, -9.1703e-04,  1.6809e-03, -4.4980e-03,
             3.3514e-03, -3.8682e-03, -2.7354e-03,  4.7370e-03, -5.7496e-03,
             2.5621e-03, -1.6336e-03, -1.2800e-03, -2.6159e-03, -5.5076e-03,
            -5.1836e-03,  2.8190e-03,  4.8737e-03, -6.9832e-04, -4.3278e-03,
            -3.7068e-04, -3.7078e-03, -8.1038e-03, -2.5190e-03,  4.6775e-03,
            -2.4334e-03, -3.7394e-03, -1.2748e-03,  3.9921e-03,  2.5756e-03,
             2.6131e-03,  1.1783e-03, -6.7415e-03, -6.1974e-03, -7.0965e-04,
            -3.5731e-03,  3.3572e-03, -3.0732e-04, -3.8837e-03, -4.9384e-03,
             1.0068e-03, -3.7268e-03,  3.1904e-03,  1.3117e-03,  4.6962e-03,
             1.9149e-03,  2.5360e-03,  3.0977e-03,  5.6199e-03, -4.6367e-03,
            -1.8896e-03,  1.5227e-03, -5.1942e-03,  5.8826e-04, -3.5053e-03,
            -1.4231e-04,  5.5008e-03, -5.1090e-03,  7.0833e-03,  5.1202e-04,
             2.8039e-03, -3.5043e-03,  4.4289e-03, -4.0225e-03, -6.2141e-03,
            -9.3256e-04, -6.5474e-03,  2.4153e-04, -2.5446e-03, -2.6585e-03,
            -1.9035e-03,  3.0044e-03, -3.2942e-03, -5.5797e-03,  2.6263e-03,
            -1.0658e-03, -1.1529e-04, -4.6982e-03, -2.9916e-03,  3.5620e-03,
             7.2294e-03, -3.0563e-03,  8.5442e-04, -3.1934e-03, -3.5860e-03,
             6.0537e-04, -1.5505e-05,  3.2507e-03, -3.6615e-04,  1.4924e-03,
             2.0032e-03,  4.3500e-03,  2.4332e-03,  4.1175e-03,  1.6403e-03,
            -4.6859e-03,  4.8007e-04, -7.0142e-03,  2.1541e-03, -2.9608e-03,
            -2.8193e-03, -5.9687e-03, -1.3332e-03,  6.6066e-03,  2.1767e-03,
            -4.7907e-03, -1.3376e-03, -4.6700e-03,  8.7034e-03, -1.3741e-03,
             9.2835e-04,  2.7267e-03,  3.5604e-03,  2.0370e-03, -7.8065e-04,
            -2.8343e-03, -3.7868e-05,  2.2917e-03, -2.3071e-03,  4.2288e-03,
            -6.6490e-03, -3.3571e-03, -1.5313e-04, -2.9223e-03,  6.9711e-03,
             1.0276e-03,  1.9773e-03, -3.7289e-03, -9.4421e-03, -3.4700e-03,
            -4.3431e-03,  4.6294e-03, -1.7116e-05, -2.9040e-03, -3.6404e-03,
             1.0972e-03, -2.0989e-03, -9.9074e-05, -4.5303e-03,  3.9022e-03,
            -9.3448e-04, -2.8916e-03,  6.1486e-04, -2.0581e-03,  1.9634e-03,
            -3.6450e-03, -3.7734e-03,  4.7753e-03, -2.0537e-03, -5.6879e-03,
             6.1945e-03,  2.2426e-03,  3.9778e-03,  3.9132e-04,  2.5237e-03,
            -3.4166e-03,  5.7902e-03, -3.7985e-03, -6.0184e-03, -3.4301e-03,
            -2.5473e-03, -1.0651e-03, -4.1023e-03, -1.2727e-03,  9.6251e-04,
             1.3904e-04,  2.2970e-03, -5.0236e-03,  3.5743e-03,  3.5472e-03,
            -2.0900e-03,  4.9596e-03, -4.9601e-03,  6.4874e-03,  3.0922e-03,
            -3.4291e-03,  7.0884e-04, -1.3523e-03, -4.9018e-03,  6.3482e-04,
            -4.9905e-03,  2.4724e-03, -2.7708e-03, -4.1508e-03, -5.9863e-03,
            -3.1012e-04, -3.6736e-03, -2.7308e-03,  3.6665e-03, -1.6742e-03,
            -2.1726e-03,  1.0783e-03, -3.6468e-03, -8.8143e-04, -4.6692e-03,
            -3.4079e-03, -4.6019e-03,  5.1132e-05, -5.3712e-03,  6.5391e-03,
            -7.9607e-03, -4.9854e-03, -1.0118e-03, -6.2795e-04,  1.6702e-03,
             1.8199e-04, -5.8542e-03,  2.8069e-03,  1.9518e-03,  7.6851e-03,
            -1.8264e-03, -1.0629e-03,  2.2670e-03,  5.3221e-03,  5.2727e-03,
            -6.0633e-03,  6.9601e-03,  4.8215e-03,  4.1850e-05, -3.0908e-05,
            -5.2333e-04,  4.5711e-03, -6.6444e-03,  1.4458e-03,  3.3509e-03,
             1.6464e-03, -3.6089e-03, -1.8538e-03, -2.2141e-03,  2.9857e-03,
            -2.5902e-03, -1.1702e-03, -4.7838e-03,  1.4145e-03,  2.4180e-03,
            -2.5118e-03, -1.2048e-03,  4.6300e-03,  5.3292e-03, -5.1292e-03,
             3.7460e-03, -5.1509e-03,  1.4141e-03,  3.3522e-03, -2.3746e-03,
            -4.5765e-03, -2.3610e-03, -1.1832e-03, -7.2326e-03, -2.2711e-03,
             9.0293e-04,  4.3562e-03,  5.8650e-03,  6.6427e-04, -5.2263e-03,
            -7.7427e-04,  1.8121e-03, -1.3488e-03,  4.1201e-03,  6.8937e-03,
            -2.2698e-03,  6.4644e-03, -5.9418e-03, -1.8312e-03, -1.1247e-03,
            -5.0182e-03,  1.6217e-03, -2.3265e-03, -5.1708e-03, -8.0764e-04,
            -6.8422e-03, -1.0885e-03, -4.6536e-03,  3.2646e-03, -2.8515e-03,
            -4.3452e-03, -7.7435e-04,  1.2259e-03, -8.6269e-04, -1.8554e-03,
             4.5647e-03,  3.7145e-03, -8.4355e-03, -3.4337e-03,  8.8910e-05,
             3.7513e-03,  1.0517e-03, -3.1532e-03, -3.1331e-03,  2.9339e-04,
             7.5329e-03, -3.4472e-03, -2.0644e-03, -9.8422e-05, -7.5511e-03,
            -3.4629e-03, -8.8003e-04, -2.1700e-03,  1.9822e-03, -2.5383e-03,
             4.7716e-03, -7.3063e-03,  2.9904e-03,  2.2591e-03, -2.6333e-03,
             6.6211e-03,  2.7325e-03, -1.2473e-04,  7.5166e-04, -1.9293e-03,
             3.1495e-04, -1.0772e-03,  1.0714e-04, -2.8535e-03,  8.8554e-04,
            -3.1941e-04, -4.0212e-03, -6.4445e-03, -1.8603e-03, -6.1687e-04,
             5.6115e-03,  4.2017e-03,  2.2555e-03, -3.4749e-03,  5.0626e-04,
             2.3245e-04, -9.3784e-04,  5.0290e-03,  3.0940e-04,  4.8555e-03,
            -3.3787e-06, -3.8365e-03,  1.5970e-03,  3.7624e-03, -3.6584e-03,
            -1.6831e-03,  7.0722e-03,  3.2046e-03, -5.0180e-03, -3.4969e-03,
            -1.4666e-03, -7.8451e-03,  1.6311e-03, -4.9027e-03, -1.9275e-03,
            -1.2958e-03, -3.8805e-03,  2.5859e-04,  3.3558e-04,  6.6623e-03,
             3.4600e-03, -7.5133e-03, -1.9788e-04, -1.1696e-03, -7.6580e-03,
            -6.3148e-03, -1.8480e-03,  2.9675e-03, -2.6992e-04,  2.6299e-03,
             3.5601e-03, -7.2615e-03, -4.0368e-03, -1.5048e-03,  1.5199e-03,
             2.4896e-04, -8.3634e-05,  3.2043e-03,  6.1538e-03, -1.0206e-03,
             2.0250e-04,  6.5058e-03, -6.4812e-04, -3.2648e-03,  2.2847e-03,
             4.3792e-03,  7.1255e-03, -4.9991e-03,  3.1483e-03,  8.1778e-03,
             4.4621e-03,  2.7427e-03], device='cuda:0'), tensor([ 2.9266e-02,  1.9604e-02,  7.3191e-02,  4.0470e-02,  6.3105e-03,
             1.4130e-02,  4.2311e-02,  6.5825e-02,  2.1575e-02, -2.6634e-02,
            -8.4416e-02, -6.8192e-02, -4.0233e-02, -3.9119e-02, -6.1423e-02,
             5.1264e-03, -1.5217e-03, -8.7141e-02, -2.2648e-02, -3.6711e-02,
            -1.4640e-02, -4.5751e-02, -2.9054e-02, -2.5084e-02, -3.4787e-02,
            -5.1488e-02,  4.3208e-02,  2.9525e-02, -5.7947e-02,  1.0509e-01,
            -3.2763e-02,  7.2743e-02, -9.6812e-02, -2.0365e-02,  9.4584e-04,
            -7.5155e-02, -2.4134e-02,  3.3644e-02,  2.2866e-03, -3.5486e-03,
             4.2205e-02,  7.1820e-02, -5.8204e-02, -5.2165e-02, -5.0368e-02,
            -1.0436e-01,  3.6968e-02, -7.7493e-02, -2.0537e-02, -4.1254e-02,
             4.3018e-02, -9.8436e-02,  6.1240e-02, -3.7124e-03, -5.2769e-02,
            -6.7276e-02, -2.5292e-02,  3.6493e-02,  1.5638e-02,  7.2422e-03,
             9.6131e-03, -1.9798e-02, -2.4852e-02,  1.4559e-03,  5.2314e-03,
            -4.0833e-02, -4.4964e-02,  3.7891e-02, -5.5990e-02, -2.3204e-02,
            -7.0031e-03,  3.0227e-02, -2.5805e-02, -5.0610e-02, -1.6393e-01,
             9.8286e-03, -3.5812e-03,  3.2242e-02,  2.5743e-02,  3.6529e-02,
             2.0598e-03,  8.8765e-02,  7.1278e-02,  3.9823e-02,  3.3278e-02,
            -3.3269e-02, -3.7071e-02,  1.9033e-02, -1.4694e-02,  3.6782e-02,
             5.7621e-02,  4.0118e-03,  6.9233e-02,  3.3135e-03, -4.7083e-02,
             2.1420e-02, -1.6569e-02, -5.3740e-02,  6.7239e-02, -1.6742e-02,
            -2.1698e-02, -6.8688e-02, -1.2596e-02, -1.2614e-01, -3.8628e-02,
            -9.3657e-03,  6.2692e-02,  1.8579e-02, -5.2706e-02,  1.7397e-02,
            -5.2837e-02, -8.7347e-03, -3.5185e-02, -2.7680e-02, -1.1473e-03,
            -4.6510e-02, -6.7450e-03, -3.7548e-02,  5.3523e-02, -1.2464e-03,
             2.5984e-02, -2.4758e-02, -2.5923e-02,  3.8857e-02,  5.3308e-02,
             7.1394e-02,  6.1359e-02, -6.8861e-02,  6.2815e-02, -7.1213e-04,
             2.4883e-02, -4.4154e-02,  1.9425e-02, -2.6996e-02, -7.2200e-02,
            -1.4332e-02,  7.8670e-03,  2.6061e-02, -3.9178e-02, -6.3025e-02,
            -4.3769e-03, -4.0285e-02, -2.7928e-02,  3.9982e-02,  2.3567e-02,
            -7.8410e-03,  7.1261e-02,  4.6431e-02,  4.6529e-02, -1.0340e-04,
            -3.7422e-03,  7.2721e-03,  3.8496e-02, -3.3444e-02, -6.6841e-02,
            -1.2154e-02, -6.3186e-02, -3.5385e-02,  4.5176e-02, -3.2815e-02,
             1.9979e-02, -3.0835e-02, -2.0068e-02,  4.3450e-02, -2.8510e-02,
             2.4528e-02, -1.1989e-02,  5.4181e-02,  1.3351e-02,  3.7262e-02,
            -5.4133e-02,  4.4162e-04,  3.2876e-02, -4.9788e-02,  3.6522e-02,
             2.0182e-02,  6.4367e-02,  2.3907e-02, -5.7212e-02, -5.5801e-02,
            -3.2901e-02,  1.0095e-01,  7.2469e-02,  3.2173e-02,  6.4561e-02,
            -3.5807e-02,  5.0878e-02, -1.0112e-01,  3.0290e-02,  6.8844e-02,
            -1.1827e-02, -6.4715e-02,  2.6422e-02, -6.4505e-02, -2.6529e-02,
             1.8196e-02, -5.5309e-03,  2.1013e-02,  4.3145e-02, -7.9663e-02,
            -8.2500e-03, -5.2665e-02,  2.4433e-02,  3.1344e-03, -1.9354e-02,
             5.2550e-02, -3.9881e-02,  1.9022e-02, -1.7071e-02, -9.6024e-02,
             5.8924e-02, -5.5854e-02,  4.0312e-02,  4.6649e-02,  7.3550e-02,
            -4.5994e-02,  5.8567e-03, -8.8510e-02, -3.1080e-02, -8.6699e-02,
            -5.8078e-02,  2.3269e-02, -9.0518e-02,  9.0677e-02, -5.5050e-02,
             1.4886e-01,  8.2282e-02, -8.2819e-02, -1.0054e-01,  1.1048e-02,
            -7.2651e-02,  3.4368e-02, -1.2541e-02,  5.4069e-02, -7.0576e-03,
             1.6096e-02, -7.4243e-02, -6.1706e-02, -1.3370e-02, -4.4311e-02,
             5.9100e-02,  2.8574e-03, -1.4011e-02,  1.3736e-02,  5.0613e-02,
            -1.8811e-02,  5.6978e-03,  9.8886e-02,  8.3116e-03, -3.1431e-02,
             1.3228e-03,  4.3681e-02,  4.1504e-02, -3.1287e-02,  3.8262e-02,
             4.6720e-02,  5.5237e-02, -3.5829e-03, -3.8130e-02,  9.2721e-03,
            -1.5470e-01,  4.4310e-02, -2.5394e-04,  8.6069e-02, -6.2606e-02,
            -6.2688e-02,  5.9288e-02,  1.6623e-03,  5.7170e-02,  2.4138e-02,
             1.0294e-02, -7.4923e-02,  2.1190e-02,  1.8708e-02,  1.0895e-02,
            -1.0552e-02, -7.7499e-02,  9.6863e-02,  3.6210e-02,  4.6553e-02,
             1.6736e-02,  3.4793e-03,  3.1221e-02,  6.0272e-02,  6.4010e-04,
            -8.3158e-02, -3.1264e-02, -1.4488e-02,  4.6291e-02,  9.8476e-03,
             3.8969e-02,  2.3592e-03,  4.7906e-02, -4.6783e-02, -4.2213e-02,
            -6.5180e-02,  2.0572e-02,  4.1863e-02, -1.9125e-02, -1.0150e-01,
            -3.7132e-02,  1.6232e-02,  8.3258e-02, -8.4431e-04,  2.4225e-02,
            -4.5842e-02,  6.1654e-02,  4.1571e-03,  6.1464e-02,  4.2046e-03,
            -3.4362e-02, -3.5480e-02, -1.0818e-01, -1.3904e-02,  6.8247e-02,
            -1.3479e-02,  5.3957e-02,  3.6734e-02,  4.5947e-03, -2.3509e-02,
            -8.8684e-04,  6.4579e-02,  3.8303e-02, -3.6039e-02,  4.9322e-02,
            -2.5867e-02,  7.0290e-02,  1.5880e-02, -2.3479e-02,  1.1643e-01,
             6.1977e-04,  3.5471e-05,  9.8971e-03,  5.9975e-02, -8.8453e-02,
             8.5727e-02,  2.5786e-02,  1.7382e-02,  3.9698e-02,  7.0399e-02,
             2.0185e-02, -4.7881e-02,  1.1222e-02, -9.3099e-02,  1.9619e-02,
             4.3460e-02, -4.1052e-02, -2.2520e-02, -2.2635e-02, -2.9422e-02,
             3.0507e-03, -6.8542e-03,  8.3593e-02, -1.2500e-02, -1.8240e-02,
            -3.3302e-03, -5.3549e-02,  5.9071e-02, -1.8904e-02,  5.3757e-02,
            -3.3894e-03,  1.5498e-02,  2.4229e-03, -2.8795e-03, -7.4174e-02,
            -2.8904e-02, -3.4225e-02, -4.2495e-02, -3.9095e-02,  2.3712e-02,
            -4.5922e-02, -2.5507e-02, -6.6362e-03,  1.1946e-02,  5.7384e-02,
             2.4752e-03, -1.7141e-02, -2.9270e-02, -6.3800e-02, -6.7637e-02,
            -7.4111e-02,  5.3867e-02,  7.1502e-02,  4.0338e-02,  1.6047e-02,
             1.1875e-02,  6.1981e-02,  7.2944e-02, -5.8359e-02,  6.3019e-02,
            -4.1621e-02, -4.0921e-02,  1.4186e-02,  1.0884e-01, -2.1118e-02,
             6.6439e-03, -1.2708e-02,  4.8100e-03,  7.0121e-02, -2.8805e-02,
             5.2721e-03,  6.5302e-02,  4.6728e-04, -1.1483e-01, -1.0221e-02,
             1.3519e-02,  1.6354e-02, -1.2749e-02, -3.7750e-02,  1.0038e-01,
             2.6220e-02, -5.5796e-02, -3.0849e-02,  2.3049e-02,  4.9174e-02,
             1.2585e-01,  3.1937e-03, -4.7947e-02, -2.7790e-02, -3.4260e-02,
            -4.8420e-02, -4.1243e-02,  6.9628e-02,  1.9886e-02, -2.6308e-02,
            -5.1269e-02, -5.6632e-02,  4.2628e-02,  5.0521e-02, -1.6800e-02,
            -1.2166e-01,  1.9066e-02, -1.8109e-02, -2.3838e-02, -7.2598e-02,
            -7.9241e-02,  3.9061e-02,  1.4182e-02,  2.4509e-02,  2.4297e-02,
            -7.6681e-02, -9.6821e-03, -2.5949e-02, -5.6209e-02,  4.1290e-02,
             2.3681e-02,  2.4660e-02, -2.9203e-03,  4.5415e-02,  3.6780e-02,
            -2.1301e-02, -1.1925e-02,  5.3408e-02,  3.1702e-03,  3.7197e-02,
             2.6603e-02,  2.4465e-02,  3.1103e-02,  4.1573e-02, -7.1140e-02,
             1.3058e-02, -5.7184e-02, -2.6181e-02, -5.8899e-02, -1.9199e-02,
             1.2735e-02, -2.4638e-02,  2.9557e-02, -4.2086e-02, -1.2358e-02,
             4.0275e-02,  1.1090e-02,  3.9624e-02,  1.0299e-02,  9.9786e-02,
             2.6026e-02,  1.2548e-01,  1.2778e-01, -3.8073e-02, -3.6304e-02,
             1.7441e-04,  6.1777e-02,  7.5222e-02,  3.5223e-02,  1.3180e-02,
             4.0593e-02, -6.4993e-02, -5.6531e-03,  4.9200e-02, -7.7678e-03,
             3.1387e-02, -7.1530e-03, -4.0115e-02,  5.4024e-02, -3.1771e-02,
            -1.5580e-02,  2.8979e-02, -6.0992e-02,  2.7479e-02,  1.7934e-03,
             6.0230e-02, -2.2776e-02, -3.4095e-04,  1.3992e-02,  5.2726e-03,
             2.0902e-02,  7.2112e-02,  1.4398e-03,  6.4597e-03, -4.9511e-02,
            -6.5775e-02, -3.2125e-03], device='cuda:0'), tensor([[-0.0446,  0.0264,  0.0425,  ...,  0.0256,  0.0149,  0.0147],
            [-0.0176, -0.0002, -0.0492,  ...,  0.0319,  0.0011, -0.0424],
            [-0.0372,  0.0462,  0.0082,  ...,  0.0135,  0.0210, -0.0227],
            ...,
            [ 0.0169, -0.0520, -0.0318,  ..., -0.0283, -0.0390, -0.0253],
            [ 0.0506,  0.0169, -0.0159,  ..., -0.0164, -0.0347, -0.0209],
            [ 0.0034, -0.0378,  0.0618,  ...,  0.0389, -0.0823,  0.0355]],
           device='cuda:0'), tensor([ 0.0234,  0.0143,  0.0165, -0.0386, -0.0189,  0.0121, -0.0061, -0.0152,
             0.0071,  0.0229, -0.0301,  0.0061, -0.0234,  0.0309], device='cuda:0')])

 

这个模型最后的效果只有0.85的分数，甚至还不如之前尝试的TF-IDF+LR尝试出的最好为0.919的成绩，不过也可能是因为训练的epoch确实比较少的原因。



即使是前面的LR也有人训练出0.93的成绩，但是我目前却只能找到0.919左右，说明还是有一些小技巧我需要去掌握的。