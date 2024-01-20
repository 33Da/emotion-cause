import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys, warnings, time

sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
from data_loader import *
from model import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *

import random

torch.cuda.set_device(0)


def main(configs, fold_id):
    random.seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    print("USE GPU:", torch.cuda.is_available(), " Index:", torch.cuda.current_device())

    print("加载数据")
    train_loader = build_train_data(configs, fold_id=fold_id)


    test_pair = get_ecpair(configs, fold_id=fold_id, data_type='test')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')

    print("加载模型")
    model = Network(configs).to(DEVICE)

    print(configs.warmup_proportion, configs.l2_bert, configs.l2_other, configs.lr)


    paramsbert = []
    paramsbert0reg = []
    paramsothers = []

    for name, parameters in model.named_parameters():
        if not parameters.requires_grad:
            continue
        if 'bert' in name:
            if '.bias' in name or 'LayerNorm.weight' in name:
                paramsbert0reg += [parameters]
            else:
                paramsbert += [parameters]
        else:
            paramsothers += [parameters]

    params = [dict(params=paramsbert, weight_decay=configs.l2_bert),
              dict(params=paramsothers, lr=1e-4, weight_decay=configs.l2_other),
              dict(params=paramsbert0reg, weight_decay=0.0)]
    optimizer = AdamW(params, lr=configs.lr, weight_decay=0)

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_steps_all)

    model.zero_grad()
    max_ec, max_e, max_c = (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)
    metric_ec, metric_e, metric_c = (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)
    early_stop_flag, best_epoch = 0, 0
    for epoch in range(1, configs.epochs + 1):
        if epoch == 10 and metric_ec[2] <= 0.5:
            break
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, y_causes_b_1, y_polarity_b, y_token_fact, y_token_mask_b = batch

            pred_e, static_query, doc_sent, fact_static_query = model(bert_token_b, y_token_fact,
                                                                                bert_masks_b,y_token_mask_b,
                                                                                bert_clause_b, doc_len_b,
                                                                                y_causes_b)
            loss = model.loss_pre(pred_e, y_polarity_b, y_emotions_b, y_causes_b, static_query,
                                  fact_static_query, doc_sent, doc_len_b)
            loss = loss / configs.gradient_accumulation_steps
            if train_step <= 50:
                print('epoch:', epoch, loss.item())

            loss.backward()
            if train_step % configs.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 10)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        with torch.no_grad():
            model.eval()
            print('epoch :', epoch, 'eval is begining')
            if configs.split == 'split10_emo':
                a, b, c = inference_one_epoch(configs, test_loader, model, test_pair, 1)
                print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(c[2]), float_n(c[0]),
                                                               float_n(c[1])))
                print(
                    'F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(a[2]), float_n(a[0]),
                                                             float_n(a[1])))
                print(
                    'F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(b[2]), float_n(b[0]),
                                                             float_n(b[1])))

                if a[2] > metric_e[2]:
                    metric_e = a

                if b[2] > metric_c[2]:
                    early_stop_flag = 1
                    metric_c = b

                if c[2] > metric_ec[2]:
                    early_stop_flag = 1
                    metric_ec = c
                    best_epoch = epoch
                else:
                    early_stop_flag += 1
                    if early_stop_flag > 5 and epoch >= 5:
                        print("best epoch:", best_epoch)
                        return metric_ec, metric_e, metric_c

    return metric_ec, metric_e, metric_c


def inference_one_epoch(configs, batches, model, pair, epoch):
    for batch in batches:
        doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
        bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, y_causes_b_1, y_polarity_b,y_token_fact, y_token_mask_b = batch
        pred_pair1, pred_e, pred_c = model.inference(bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b,
                                                     doc_len_b, y_causes_b, doc_id_b)

        pred_pair1 = lexicon_based_extraction(doc_id_b, pred_pair1)

        (p_e, r_e, f_e), (p_c, r_c, f_c) = eval_func(pair, list(pred_pair1))

        p_ec, r_ec, f1_ec = prf_2nd_step(pair, pred_pair1)

    return (p_e, r_e, f_e), (p_c, r_c, f_c), (p_ec, r_ec, f1_ec)


def lexicon_based_extraction(doc_ids, couples_pred):
    emotional_clauses = read_b(os.path.join(DATA_DIR, SENTIMENTAL_CLAUSE_DICT))

    couples_pred_filtered = []
    for i, couples_pred_i in enumerate(couples_pred):
        emotional_clauses_i = emotional_clauses[str(int(couples_pred_i / 10000))]
        if int((couples_pred_i % 10000) / 100) in emotional_clauses_i:
            couples_pred_filtered.append(couples_pred_i)

    return couples_pred_filtered


if __name__ == '__main__':
    configs = Config()

    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 20
    else:
        print('Unknown data split.')
        exit()

    metric_folds = {'ecp': [], 'emo': [], 'cau': []}

    for fold_id in range(1, n_folds + 1):
        print('===== fold {} ====='.format(fold_id))
        metric_ec, metric_e, metric_c = main(configs, fold_id)

        metric_folds['ecp'].append(metric_ec)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)
        print("##" * 10)
        print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]),
                                                       float_n(metric_ec[1])))
        print(
            'F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
        print(
            'F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))

    metric_ec = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    metric_e = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    metric_c = np.mean(np.array(metric_folds['cau']), axis=0).tolist()

    print('===== Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]), float_n(metric_ec[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))

    write_b({'ecp': metric_ec, 'emo': metric_e, 'cau': metric_c},
            '{}_{}_metrics.pkl'.format(time.time(), configs.split))

