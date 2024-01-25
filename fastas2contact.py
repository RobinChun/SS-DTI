import numpy as np
import torch
import pickle as pkl
import re
import os
import esm


torch.set_grad_enabled(False)


def load_seqs(fn, label=1):
    """
    :param fn: source file name in fasta format
    :param tag: label = 1(positive, AMPs) or 0(negative, non-AMPs)
    :return:
        ids: name list
        seqs: peptide sequence list
        labels: label list
    """
    ids = []
    seqs = []
    t = 0
    # Filter out some peptide sequences
    pattern = re.compile('[^ARNDCQEGHILKMFPSTWYVUX]')
    i = 0
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                # t = line.replace('|', '_')
                # t = t.replace('>', '')
                t = i
                i = i + 1
            elif len(pattern.findall(line)) == 0:
                seqs.append(line)
                ids.append(t)
                t = 0
            else:
                print(i)
                print(line)
    if label == 1:
        labels = np.ones(len(ids))
    else:
        labels = np.zeros(len(ids))
    # print(ids)
    return ids, seqs, labels


def to_esm_format_seqs(names, seqs):
    # print("name:",names)
    res_seqs = {}
    for i in range(len(names)):
        res_seqs[names[i]] = ('', seqs[i])
    return res_seqs


def predict_contact(sequences, cuda=False):
    # ====================== ESM-2 Predictions
    esm2, esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # batch_converter = esm2_alphabet.get_batch_converter()
    # esm2, esm2_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    # esm2, esm2_alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    # esm2, esm2_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm3, esm3_alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    esm2.eval()
    if cuda:
        esm2 = esm2.cuda()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()

    esm3.eval()
    esm3_batch_converter = esm3_alphabet.get_batch_converter()

    # result
    esm2_predictions = {}
    esm2_results = []
    # inputs:tuple ==>  ('_desc', 'seq')
    no = 0
    invalid = 0
    for name, inputs in sequences.items():
        print("\r", "predict: NO." + str(no), end='', flush=True)
        no += 1
        # esm2_batch_strs:List => ['seq']

        esm2_batch_labels, esm2_batch_strs, esm2_batch_tokens = esm2_batch_converter([inputs])
        if len(inputs[1]) > 6000:
            out = np.zeros((1000, 1000), dtype=float)
            invalid += 1
        elif cuda and len(inputs[1]) > 1000:
            esm2_batch_tokens = esm2_batch_tokens.to(next(esm3.parameters()).device)
            out = esm3.predict_contacts(esm2_batch_tokens)[0].cpu().numpy()
        else:
            esm2_batch_tokens = esm2_batch_tokens.to(next(esm2.parameters()).device)
            out = esm2.predict_contacts(esm2_batch_tokens)[0].cpu().numpy()
        esm2_predictions[name] = out

    print(f"\ninvalid sequences: {invalid}\n")

    return esm2_predictions


if __name__ == '__main__':
    """generate txt file for esm prediction"""
    # DATASET = 'celegans_data'
    DATASET = 'Davis'
    # DATASET = 'DrugBank'
    filepath = f'./data/{DATASET}.txt'
    with open(filepath, 'r') as f:
        raw_data = f.read().strip().split('\n')

    if not os.path.exists('./dataset/'):
        os.makedirs('./dataset/')
    dumppath = f"./dataset/{DATASET}_pos.txt"
    with open(dumppath, 'w') as f:
        sequence_list = list()
        i = 0
        for idx, parts in enumerate(raw_data):
            items = parts.strip().split(' ')
            if len(items) == 3:
                smile, sequence, label = items
            else:
                _, _, smile, sequence, label = items
            if sequence in sequence_list:
                continue
            else:
                sequence_list.append(sequence)
                f.write(f">{i}\n{sequence}\n")
                i += 1

    print(f"{len(sequence_list)} sequences")

    # base_name = "Davis_pos"
    # base_name = 'celegans_data_pos'
    base_name = f'{DATASET}_pos'
    # cuda = False
    cuda = True

    fasta_fname = "./dataset/{}.txt".format(base_name)
    contact_dump_path = "./dataset/{}_contact.pkl".format(base_name)

    ids, seqs, labels = load_seqs(fn=fasta_fname)
    print("ids:", len(ids), ids)
    print("seqs:", len(seqs))
    # {'name': ('_desc', 'seq'), ...}
    sequences = to_esm_format_seqs(ids, seqs)
    # print("sequence:",sequences)

    esm2_predictions = predict_contact(sequences, cuda=cuda)

    pkl.dump(esm2_predictions, open(contact_dump_path, "wb"))
