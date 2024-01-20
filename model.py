# -*-coding:GBK -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel
from transformers import BertTokenizer



class Attention_Layer(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, query, key):

        Q = self.Q_linear(query)
        K = self.K_linear(key).permute(0, 2, 1)
        V = self.V_linear(key)

        alpha = torch.matmul(Q, K)

        alpha = F.softmax(alpha, dim=2)
        out = torch.matmul(alpha, V)

        return out


def inverse_predictive_loss(feature1, feature2):
    return F.mse_loss(feature1, feature2) + F.mse_loss(feature2, feature1)


class Similarity(nn.Module):
    def __init__(self, temp):
        super(Similarity, self).__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        sim = self.cos(x, y) / self.temp
        return sim


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 200)

        self.lstm1 = nn.LSTM(200, 100, bidirectional=True, batch_first=True)

        self.cause_fc = nn.Linear(600, 2)
        self.polarity_fc = nn.Linear(600, 3)

        self.query = nn.Parameter(torch.randn(768), requires_grad=True)
        self.query_linear = nn.Linear(768, 768)
        self.att_layer = Attention_Layer(200)

        self.fc5 = nn.Linear(768, 1)

        self.static_query_linear = nn.Linear(768, 768)
        self.doc_sents_linear = nn.Linear(768, 768)
        self.fact_static_query_linear = nn.Linear(768, 768)
        self.sim = Similarity(0.05)

    def forward(self, bert_token_b, y_token_fact, bert_masks_b,y_token_mask_b,
                bert_clause_b, doc_len, y_causes_b):

        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_masks_b.to(DEVICE))

        bert_output_fact = self.bert(input_ids=y_token_fact.to(DEVICE),
                                attention_mask=y_token_mask_b.to(DEVICE))


        doc_sents_h_fact = self.batched_index_select(bert_output_fact, bert_clause_b.to(DEVICE))
        fact_static_query = doc_sents_h_fact[:, 0, :]
        fact_static_query = fact_static_query.unsqueeze(1)

        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
        doc_sents_h1 = doc_sents_h[:, 1:, :]
        static_query = doc_sents_h[:, 0, :]
        static_query = static_query.unsqueeze(1)
        query = self.query_linear(static_query)

        doc_sents_h2 = torch.cat([query, doc_sents_h1], dim=1)

        doc_sents_h3, _ = self.lstm1(self.fc1(doc_sents_h2))

        query_lstm = doc_sents_h3[:, 0:1, :].expand_as(doc_sents_h3[:, 1:, :])

        doc_lstm = doc_sents_h3[:, 1:, :]
        query_att = self.att_layer(doc_lstm, query_lstm)

        doc_sents_h4 = torch.cat((query_att, doc_lstm, torch.abs(query_att - doc_lstm)), 2)

        pred_list = []

        for batch_index in range(y_causes_b.shape[0]):
            if y_causes_b[batch_index] == 0:
                pred = self.polarity_fc(doc_sents_h4[batch_index])
            else:
                pred = self.cause_fc(doc_sents_h4[batch_index])
            pred_list.append(pred)

        static_query = self.static_query_linear(static_query)
        doc_sents_h1 = self.doc_sents_linear(doc_sents_h1)
        fact_static_query = self.fact_static_query_linear(fact_static_query)

        return pred_list, static_query, doc_sents_h1, fact_static_query

    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        doc_sents_h = torch.zeros(bert_clause_b.size(0), bert_clause_b.size(1) + 1, hidden_state.size(2)).cuda()
        for i in range(doc_sents_h.shape[0]):
            for j in range(doc_sents_h.shape[1]):
                if j == doc_sents_h.shape[1] - 1:
                    hidden = hidden_state[i, bert_clause_b[i, j - 1]:, :]
                    weight = F.softmax(self.fc5(hidden), 0)
                    hidden = torch.mm(hidden.permute(1, 0), weight).squeeze(1)
                    doc_sents_h[i, j, :] = hidden
                elif bert_clause_b[i, j] != 0:
                    if j == 0:
                        hidden = hidden_state[i, 0:bert_clause_b[i, j], :]
                        weight = F.softmax(self.fc5(hidden), 0)
                        hidden = torch.mm(hidden.permute(1, 0), weight).squeeze(1)
                    else:
                        hidden = hidden_state[i, bert_clause_b[i, j - 1]:bert_clause_b[i, j], :]
                        weight = F.softmax(self.fc5(hidden), 0)
                        hidden = torch.mm(hidden.permute(1, 0), weight).squeeze(1)
                    doc_sents_h[i, j, :] = hidden
                else:
                    hidden = hidden_state[i, bert_clause_b[i, j - 1]:, :]
                    weight = F.softmax(self.fc5(hidden), 0)
                    hidden = torch.mm(hidden.permute(1, 0), weight).squeeze(1)
                    doc_sents_h[i, j, :] = hidden
                    break

        return doc_sents_h

    def loss_pre(self, pred_e, y_polarity_b, y_emotions_b, y_causes_b, static_query, fact_static_query, doc_sent, source_length):

        pred_p, prec_c, label_p, label_cause = [], [], [], []
        source_length_p, source_length_c, doc_sent_emo,doc_sent_cause = [], [], [],[]
        sim_emo_label,sim_cause_label = [],[]
        loss_c, loss_p = 0, 0
        st_sim_emo_loss, st_sim_cause_loss, dy_sim_loss,emo_inverse_loss,cause_inverse_loss = 0, 0, 0, 0, 0
        static_query_emo,static_query_cause = [],[]

        for index, y in enumerate(y_causes_b):
            if y == 0:
                pred_p.append(pred_e[index])
                label_p.append(y_polarity_b[index].tolist())
                source_length_p.append(source_length[index])
                doc_sent_emo.append(doc_sent[index])
                sim_emo_label.append(y_emotions_b[index])
                static_query_emo.append(static_query[index])
            else:
                prec_c.append(pred_e[index])
                label_cause.append(y_polarity_b[index].tolist())
                source_length_c.append(source_length[index])
                temp = torch.cat([doc_sent[index][:source_length[index]],fact_static_query[index].squeeze(1),doc_sent[index][source_length[index]:]],dim=0)
                doc_sent_cause.append(temp)
                label = y_polarity_b[index].tolist()[:source_length[index]] + [0] + y_polarity_b[index].tolist()[source_length[index]:]
                sim_cause_label.append(np.array(label))
                static_query_cause.append(static_query[index])


        if len(pred_p) != 0:
            pred_p = torch.stack(pred_p, dim=0).to(DEVICE)
            label_p = torch.LongTensor(label_p).to(DEVICE)

            packed_y = torch.nn.utils.rnn.pack_padded_sequence(pred_p.permute(1, 0, 2), list(source_length_p),
                                                               enforce_sorted=False).data
            target_ = torch.nn.utils.rnn.pack_padded_sequence(label_p.permute(1, 0), list(source_length_p),
                                                              enforce_sorted=False).data
            loss_p = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y), target_)

            doc_sent_emo = torch.stack(doc_sent_emo)
            static_query_emo = torch.stack(static_query_emo)

            st_query_emo_sim = self.sim(static_query_emo, doc_sent_emo)
            sim_emo_label = torch.LongTensor(sim_emo_label).to(DEVICE)
            sim_emo_label = torch.max(sim_emo_label, dim=-1)[1]


            for batch in range(len(pred_p)):
                st_sim_emo_loss += F.cross_entropy(st_query_emo_sim[batch][:source_length_p[batch]],
                                                   sim_emo_label[batch])
            st_sim_emo_loss /= len(pred_p)

        if len(prec_c) != 0:
            prec_c = torch.stack(prec_c, dim=0).to(DEVICE)
            label_cause = torch.LongTensor(label_cause).to(DEVICE)

            packed_y = torch.nn.utils.rnn.pack_padded_sequence(prec_c.permute(1, 0, 2), list(source_length_c),
                                                               enforce_sorted=False).data
            target_ = torch.nn.utils.rnn.pack_padded_sequence(label_cause.permute(1, 0), list(source_length_c),
                                                              enforce_sorted=False).data
            loss_c = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y), target_)

            doc_sent_cause = torch.stack(doc_sent_cause)
            static_query_cause = torch.stack(static_query_cause)


            st_query_cause_sim = self.sim(static_query_cause, doc_sent_cause)
            sim_cause_label = torch.LongTensor(sim_cause_label)
            sim_cause_label = torch.max(sim_cause_label, dim=-1)[1].to(DEVICE)

            for batch in range(len(prec_c)):
                st_sim_emo_loss += F.cross_entropy(st_query_cause_sim[batch][:source_length_c[batch] + 1],
                                                   sim_cause_label[batch])
            st_sim_cause_loss /= len(prec_c)



        loss = loss_p + loss_c + st_sim_cause_loss + st_sim_emo_loss

        return loss

    def inference(self, bert_token_b, bert_segment_b, bert_masks_b,
                  bert_clause_b, doc_len, y_causes_b, doc_id):

        doc_ids = list(doc_id)
        doc_ids_2 = []
        for a in doc_ids:
            doc_ids_2.append(int(a))

        doc_id = list(doc_id)
        doc_ids = []
        for a in doc_id:
            doc_ids.append(int(a))

        doc_id = torch.masked_select(torch.tensor(doc_ids), (1 - torch.from_numpy(y_causes_b).bool().long()).bool())

        y_causes_b = (1 - torch.from_numpy(y_causes_b).bool().long()).unsqueeze(1).bool()

        bert_token_b = torch.masked_select(bert_token_b, y_causes_b).view(-1, bert_masks_b.shape[1])
        bert_masks_b = torch.masked_select(bert_masks_b, y_causes_b).view(-1, bert_token_b.shape[1])

        bert_clause_b = torch.masked_select(bert_clause_b, y_causes_b).view(bert_masks_b.shape[0], -1)

        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE), attention_mask=bert_masks_b.to(DEVICE))


        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
        doc_sents_h1 = doc_sents_h[:, 1:, :]
        static_query = doc_sents_h[:, 0, :]
        static_query = static_query.unsqueeze(1)
        query = self.query_linear(static_query)

        doc_sents_h2 = torch.cat([query, doc_sents_h1], dim=1)

        doc_sents_h3, _ = self.lstm1(self.fc1(doc_sents_h2))

        query_lstm = doc_sents_h3[:, 0:1, :].expand_as(doc_sents_h3[:, 1:, :])

        doc_lstm = doc_sents_h3[:, 1:, :]
        query_att = self.att_layer(doc_lstm, query_lstm)

        doc_sents_h4 = torch.cat((query_att, doc_lstm, torch.abs(query_att - doc_lstm)), 2)

        pred_e = self.polarity_fc(doc_sents_h4).argmax(2)


        pair1 = []
        pred_c = []
        for i in range(pred_e.shape[0]):
            c = bert_token_b[i].numpy().tolist().copy()
            if 0 in c:
                document = c[c.index(101, 1):c.index(0)]
            else:
                document = c[c.index(101, 1):]

            b_clause_b = bert_clause_b[i].tolist()
            tmp = []
            for z in b_clause_b:
                if z != 0:
                    tmp.append(z - 9)
                else:
                    tmp.append(0)


            for j in range(pred_e.shape[1]):
                if pred_e[i, j] > 0:
                    polarity_dict = {1: [4916, 3353], 2: [3867, 3353]}
                    polarity = polarity_dict[int(pred_e[i, j])]
                    if tmp[j] == 0:
                        continue
                    elif pred_e.shape[1] - 1 == j or tmp[j + 1] == 0:
                        emotion_cause = [101] + polarity + [2658, 2697, 2094, 1368] + document[tmp[j] + 1:-1] + [4638,
                                                                                                                 1333,
                                                                                                                 1728,
                                                                                                                 3221,
                                                                                                                 6929,
                                                                                                                 702,
                                                                                                                 136,
                                                                                                                 102]
                    else:
                        emotion_cause = [101] + polarity + [2658, 2697, 2094, 1368] + document[
                                                                                      tmp[j] + 1:tmp[j + 1] - 1] + [
                                            4638, 1333, 1728, 3221,
                                            6929, 702, 136, 102]
                    input = emotion_cause + document

                    bert_clause_b_1 = [i for i, x in enumerate(input) if x == 101]
                    bert_clause_b_1.remove(0)
                    bert_clause_b_2 = torch.tensor([bert_clause_b_1])
                    input_ids = torch.tensor([input])

                    segments_ids = []
                    segments_indices = [k for k, x in enumerate(input) if x == 101]
                    segments_indices.append(len(input))
                    for k in range(len(segments_indices) - 1):
                        semgent_len = segments_indices[k + 1] - segments_indices[k]
                        if k % 2 == 0:
                            segments_ids.extend([0] * semgent_len)
                        else:
                            segments_ids.extend([1] * semgent_len)

                    if input_ids.shape[1] > 512:
                        print(doc_id[i])
                        continue
                    bert_output = self.bert(input_ids=input_ids.to(DEVICE),
                                            attention_mask=input_ids.bool().long().to(DEVICE))

                    doc_sents_h = self.batched_index_select(bert_output, bert_clause_b_2.to(DEVICE))
                    doc_sents_h1 = doc_sents_h[:, 1:, :]
                    static_query = doc_sents_h[:, 0, :]
                    static_query = static_query.unsqueeze(1)
                    query = self.query_linear(static_query)

                    doc_sents_h2 = torch.cat([query, doc_sents_h1], dim=1)

                    doc_sents_h3, _ = self.lstm1(self.fc1(doc_sents_h2))

                    query_lstm = doc_sents_h3[:, 0:1, :].expand_as(doc_sents_h3[:, 1:, :])

                    doc_lstm = doc_sents_h3[:, 1:, :]
                    query_att = self.att_layer(doc_lstm, query_lstm)
                    doc_sents_h4 = torch.cat((query_att, doc_lstm, torch.abs(query_att - doc_lstm)), 2)


                    pred_c = F.softmax(self.cause_fc(doc_sents_h4), -1).squeeze(0)
                    for k in range(pred_c.shape[0]):
                        if pred_c[k, 1] >= 0.5:
                            pair1.append(int(doc_id[i] * 10000 + j * 100 + k + 101))

        return pair1, pred_e, pred_c
