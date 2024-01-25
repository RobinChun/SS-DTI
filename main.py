import os
from model import Predictor
from tools import label_sequence, label_smiles, graph_padding, mol_features_weighted, CHARISOSMISET, CHARPROTSET, \
    set_seed, CustomDataSet, shuffle_dataset, contact_padding
from torch.utils.data import DataLoader
from tqdm import tqdm
from hyperparameters import hyperparameter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
import pickle as pkl
from rdkit import Chem
from rdkit import RDLogger


def show_result(path, label, accuracy_list, precision_list, recall_list, auc_list, prc_list):
    accuracy_mean, accuracy_var = np.mean(accuracy_list), np.var(accuracy_list)
    precision_mean, precision_var = np.mean(precision_list), np.var(precision_list)
    recall_mean, recall_var = np.mean(recall_list), np.var(recall_list)
    auc_mean, auc_var = np.mean(auc_list), np.var(auc_list)
    prc_mean, prc_var = np.mean(prc_list), np.var(prc_list)
    print("The {} model's results:".format(label))
    with open(path + 'results.txt', 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(accuracy_mean, accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(precision_mean, precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(recall_mean, recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(auc_mean, auc_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(prc_mean, prc_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(accuracy_mean, accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(precision_mean, precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(recall_mean, recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(auc_mean, auc_var))
    print('PRC(std):{:.4f}({:.4f})'.format(prc_mean, prc_var))


def to_device(data, device):
    return [d.to(device) for d in data]


def train(model, loader, optimizer, loss_fun, device, epoch, hp):
    model.train()
    train_losses = []
    pbar = tqdm(loader, ncols=100)
    pbar.set_description(f"Epoch: {epoch} / {hp.Epoch}")
    for data in pbar:
        optimizer.zero_grad()
        data = to_device(data, device)
        label = data[-1]
        prediction = model(*data[: -1])
        loss = loss_fun(prediction, label)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        avg_loss = np.average(train_losses)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})


def test(model, loader, loss_fun, device, epoch, hp):
    model.eval()
    test_losses = []
    y, l, s = [], [], []
    pbar = tqdm(loader, ncols=100, colour='blue')
    pbar.set_description(f"Epoch: {epoch} / {hp.Epoch}")
    with torch.no_grad():
        for data in pbar:
            data = to_device(data, device)
            label = data[-1]
            predicted_scores = model(*data[: -1])
            loss = loss_fun(predicted_scores, label)
            correct_labels = label.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            y.extend(correct_labels)
            l.extend(predicted_labels)
            s.extend(predicted_scores)
            test_losses.append(loss.item())
            avg_loss = np.average(test_losses)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

    pre = precision_score(y, l)
    recall = recall_score(y, l)
    au_roc = roc_auc_score(y, s)
    tpr, fpr, _ = precision_recall_curve(y, s)
    prc = auc(fpr, tpr)
    acc = accuracy_score(y, l)
    test_loss = np.average(test_losses)
    results = 'Accuracy:{:.4f}; Precision:{:.4f}; Recall:{:.4f}; AUC:{:.4f}; PRC:{:.4f}.' \
        .format(acc, pre, recall, au_roc, prc)
    print(results)
    return y, l, s, test_loss, acc, pre, recall, au_roc, prc


def save(path, model, y, l, s, save_model=False):
    with open(path + 'true_labels.txt', 'w') as fo:
        fo.writelines('\n'.join([str(i) for i in y]))
    with open(path + 'predicted_labels.txt', 'w') as fo:
        fo.writelines('\n'.join([str(i) for i in l]))
    with open(path + 'predicted_scores.txt', 'w') as fo:
        fo.writelines('\n'.join([str(i) for i in s]))
    if save_model:
        torch.save(model, path + 'model.pt')


def collate(batch_data):
    RDLogger.DisableLog('rdApp.*')
    size = len(batch_data)
    compound_max = 100
    protein_max = 1000

    node_features = torch.zeros((size, compound_max, 34), dtype=torch.float)
    adjacency = torch.zeros((size, compound_max, compound_max), dtype=torch.float)
    compound_new = torch.zeros((size, compound_max), dtype=torch.long)
    protein_new = torch.zeros((size, protein_max), dtype=torch.long)
    labels_new = torch.zeros(size, dtype=torch.long)
    contacts = torch.zeros((size, protein_max, protein_max), dtype=torch.float)

    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()
        if len(pair) == 5:
            drug_id, protein_id, compound_str, protein_str, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
        else:
            compound_str, protein_str, label = pair[-3], pair[-2], pair[-1]
        compound_int = torch.from_numpy(label_smiles(compound_str, CHARISOSMISET, compound_max))
        compound_new[i] = compound_int
        protein_int = torch.from_numpy(label_sequence(protein_str, CHARPROTSET, protein_max))
        protein_new[i] = protein_int
        label = float(label)
        labels_new[i] = np.int32(label)

        idx = sequence_list.index(protein_str)
        contact = residue2idx[idx]
        # if len(proteinstr) != contact.shape[0]:
        #     print("protein contact map not correct!!!")
        contact = contact_padding(contact, protein_max)
        contacts[i] = contact

        mol = Chem.MolFromSmiles(compound_str)
        node_feature, adj = mol_features_weighted(mol)
        node_feature, adj = graph_padding(node_feature, adj, compound_max)
        node_features[i] = node_feature
        adjacency[i] = adj

    return contacts, node_features, adjacency, compound_new, protein_new, labels_new


if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    set_seed(SEED)

    """init hyperparameters"""
    Hp = hyperparameter()

    """Load preprocessed data."""
    # DATASET = "Davis"
    # DATASET = "human_data"
    DATASET = "celegans_data"
    # DATASET = "DrugBank"
    print(f"trained on {DATASET}")

    data_path = f'./data/{DATASET}.txt'
    with open(data_path, "r") as f:
        train_data_list = np.array(f.read().strip().split('\n'))
    print("load finished")

    contact_path = f'./dataset/{DATASET}_pos_contact.pkl'
    residue2idx = pkl.load(open(contact_path, 'rb'))

    sequence_list = list()
    for parts in train_data_list:
        items = parts.strip().split(' ')
        if len(items) == 3:
            smile, sequence, interaction = items
        else:
            _, _, smile, sequence, interaction = items
        if sequence in sequence_list:
            continue
        else:
            sequence_list.append(sequence)

    print("data shuffle")
    dataset = shuffle_dataset(train_data_list, SEED)

    if torch.cuda.is_available():
        the_device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        the_device = torch.device('cpu')
        print('The code uses CPU!!!')

    Accuracy_List, AUC_List, AUPR_List, Recall_List, Precision_List = [], [], [], [], []
    K_Fold = 5
    kf = KFold(K_Fold)
    i_fold = 0
    for train_index, test_index in kf.split(dataset):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)
        i_fold += 1

        best_Acc, best_AUC, best_AUPR, best_Recall, best_Pre = 0, 0, 0, 0, 0

        train_dataset, test_dataset = dataset[train_index], dataset[test_index]
        train_dataset = CustomDataSet(train_dataset)
        test_dataset = CustomDataSet(test_dataset)

        train_dataset_load = DataLoader(train_dataset, batch_size=Hp.Batch_size, shuffle=True, num_workers=4,
                                        collate_fn=collate)
        test_dataset_load = DataLoader(test_dataset, batch_size=Hp.Batch_size, shuffle=False, num_workers=4,
                                       collate_fn=collate)
        """ create model"""
        the_model = Predictor(Hp).to(the_device)

        """weight initialize"""
        weight_p, bias_p = [], []
        for p in the_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in the_model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        the_optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': Hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}],
            lr=Hp.Learning_rate)
        the_Loss = nn.CrossEntropyLoss()

        save_path = f"./results/{DATASET}/fold_{i_fold}/"

        """Output files."""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fold_results = save_path + f'fold_{i_fold}.txt'

        with open(fold_results, 'w') as f:
            hp_attr = '\n'.join(['%s:%s' % item for item in Hp.__dict__.items()])
            f.write(hp_attr + '\n')

        for ep in range(1, Hp.Epoch + 1):
            train(the_model, train_dataset_load, the_optimizer, the_Loss, the_device, ep, Hp)

            Y, P, S, _, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
                test(the_model, test_dataset_load, the_Loss, the_device, ep, Hp)
            if AUC_test > best_AUC:
                best_Acc, best_AUC, best_AUPR, best_Recall, best_Pre = \
                    Accuracy_test, AUC_test, PRC_test, Recall_test, Precision_test
                save(save_path, the_model, Y, P, S, save_model=True)

        AUC_List.append(best_AUC)
        Accuracy_List.append(best_Acc)
        AUPR_List.append(best_AUPR)
        Recall_List.append(best_Recall)
        Precision_List.append(best_Pre)
        with open(fold_results, 'a') as f:
            f.write(
                f"Best result: Acc: {best_Acc}, Pre: {best_Pre}, Rec: {best_Recall}, "
                f"AUC: {best_AUC}, AUPR: {best_AUPR}")

    result_path = f"./results/{DATASET}/"
    show_result(result_path, "final", Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List)
