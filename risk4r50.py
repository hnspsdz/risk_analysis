from __future__ import print_function
import os
from glob import glob
from os.path import join
import numpy as np
import time
import sys
from  Folder import ImageFolder
from torch.autograd import Variable
from tqdm import tqdm
import torch, random
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d
from PIL import Image
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *
from risk_one_rule import risk_dataset
from risk_one_rule import risk_torch_model
import risk_one_rule.risk_torch_model as risk_model
from common import config as config_risk
from scipy.special import softmax

import csv

cfg = config_risk.Configuration(config_risk.global_data_selection, config_risk.global_deep_learning_selection)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

"""Seed and GPU setting"""
seed = (int)(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.cuda.manual_seed(seed)

cudnn.benchmark = True
cudnn.deterministic = True


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def output_risk_scores(file_path, id_2_scores, label_index, ground_truth_y, predict_y):
    op_file = open(file_path, 'w', 1, encoding='utf-8')
    for i in range(len(id_2_scores)):
        _id = id_2_scores[i][0]
        _risk = id_2_scores[i][1]
        _label_index = label_index.get(_id)
        _str = "{}, {}, {}, {}".format(ground_truth_y[_label_index],
                                       predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file.write(_str + '\n')
    op_file.flush()
    op_file.close()
    return True

def prepare_data_4_risk_data():
    """
    first, generate , include all_info.csv, train.csv, val.csv, test.csv.
    second, use csvs to generate rules. one rule just judge one class
    :return:
    """
    train_data, validation_data, test_data = risk_dataset.load_data(cfg)
    return train_data, validation_data, test_data

def prepare_data_4_risk_model(train_data, validation_data, test_data):

    rm = risk_torch_model.RiskTorchModel()
    rm.train_data = train_data
    rm.validation_data = validation_data
    rm.test_data = test_data
    return rm

# --------------------------------------------------------------------------------

class Risk4r40():


    def train(  nnIsTrained, class_num, batch_size, nb_epoch, val_num, store_name, model_path=None):
        save_name = os.path.join('/home/4t/ltw/risk_val_pmg_result/', str(val_num), store_name.split('/')[-1],
                                 str(seed))
        if (not os.path.exists(save_name)):
            os.makedirs(save_name)

        # setup output
        exp_dir = save_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            os.stat(exp_dir)
        except:
            os.makedirs(exp_dir)

        use_cuda = torch.cuda.is_available()


        model_zoo = {'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152,
                         'rx50': resnext50_32x4d}
        model = model_zoo['r50'](pretrained=True).cuda()
        model.fc = nn.Linear(model.fc.in_features, 2)


        lr_begin = 0.0005
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)


        if model_path != None:
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt['model'], False)


        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,16,2)

        # ---- TRAIN THE NETWORK
        train_data, val_data, test_data = prepare_data_4_risk_data()
        risk_data = [train_data, val_data, test_data]



        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


        transformList = []
        transformList.append(transforms.Resize(256))

        transformList.append(transforms.FiveCrop(224))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_test = transforms.Compose(transformList)

        testset = MyImageFolder(
            root='/home/4t/lfy/datasets/{}/test'.format(store_name), transform=transform_test

        )
        TestDataLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=0)
        LSLLoss = LabelSmoothingLoss(class_num, 0.1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        max_test_acc=0

        for epochID in range(0, nb_epoch):


            _,train_pre,train_dis=Risk4r40.test('/home/4t/lfy/datasets/{}/train'.format(store_name),model, 1)
            _,val_pre=Risk4r40.Valtest('/home/4t/lfy/datasets/{}/val'.format(store_name),model, 1)
            _,test_pre,test_dis=Risk4r40.test('/home/4t/lfy/datasets/{}/test'.format(store_name),model, 1)



            if epochID==0:
                train_dis = train_dis.cpu().numpy().tolist()
                with open(exp_dir + '/train_dis.csv', 'w') as csvfile:
                    write = csv.writer(csvfile)
                    write.writerows(train_dis)

                test_dis = test_dis.cpu().numpy().tolist()
                with open(exp_dir + '/test_dis.csv', 'w') as csvfile:
                    write = csv.writer(csvfile)
                    write.writerows(test_dis)

                test_dis = test_pre.cpu().numpy().tolist()
                with open(exp_dir + '/test_pre.csv', 'w') as csvfile:
                    write = csv.writer(csvfile)
                    write.writerows(test_dis)

                _, test_pre2 = torch.max(test_pre.data, 1)
                test_pre2 = test_pre2.cpu().numpy().tolist()
                test_pre3 = [[str(i)] for i in test_pre2]
                with open(exp_dir + '/test_label.csv', 'w') as csvfile:
                    write = csv.writer(csvfile)
                    write.writerows(test_pre3)




            my_risk_model = prepare_data_4_risk_model(risk_data[0], risk_data[1], risk_data[2])
            train_one_pre = torch.empty((0, 1), dtype=torch.float64)
            val_one_pre = torch.empty((0, 1), dtype=torch.float64)
            test_one_pre = torch.empty((0, 1), dtype=torch.float64)


            a, _ = torch.max(train_pre, 1)
            b, _ = torch.max(val_pre, 1)
            c, _ = torch.max(test_pre, 1)

            train_one_pre = torch.cat((train_one_pre.cpu(), torch.reshape(a, (-1, 1))), dim=0).cpu().numpy()
            val_one_pre = torch.cat((val_one_pre.cpu(), torch.reshape(b, (-1, 1))), dim=0).cpu().numpy()
            test_one_pre = torch.cat((test_one_pre.cpu(), torch.reshape(c, (-1, 1))), dim=0).cpu().numpy()
            train_labels = torch.argmax(train_pre, 1).cpu().numpy()
            # np.save('train_label.npy', train_labels)
            val_labels = torch.argmax(val_pre, 1).cpu().numpy()
            # np.save('val_label', val_labels)
            test_labels = torch.argmax(test_pre, 1).cpu().numpy()
            # np.save('test_label', test_labels)


            my_risk_model.train(train_one_pre, val_one_pre, test_one_pre, train_pre.cpu().numpy(),
                                     val_pre.cpu().numpy(),
                                     test_pre.cpu().numpy(), train_labels, val_labels, test_labels, epochID)
            my_risk_model.predict(test_one_pre, test_pre.cpu().numpy(), )

            test_num = my_risk_model.test_data.data_len
            test_ids = my_risk_model.test_data.data_ids
            test_pred_y = test_labels
            test_true_y = my_risk_model.test_data.true_labels
            risk_scores = my_risk_model.test_data.risk_values

            id_2_label_index = dict()
            id_2_VaR_risk = []
            for i in range(test_num):
                id_2_VaR_risk.append([test_ids[i], risk_scores[i]])
                id_2_label_index[test_ids[i]] = i
            id_2_VaR_risk = sorted(id_2_VaR_risk, key=lambda item: item[1], reverse=True)
            if epochID == 0:
                output_risk_scores(exp_dir + '/risk_score.txt', id_2_VaR_risk, id_2_label_index, test_true_y,
                                   test_pred_y)

            id_2_risk = []
            for i in range(test_num):
                test_pred = test_one_pre[i]
                m_label = test_pred_y[i]
                t_label = test_true_y[i]
                if m_label == t_label:
                    label_value = 0.0
                else:
                    label_value = 1.0
                id_2_risk.append([test_ids[i], 1 - test_pred])
            id_2_risk_desc = sorted(id_2_risk, key=lambda item: item[1], reverse=True)
            if epochID == 0:
                output_risk_scores(exp_dir + '/base_score.txt', id_2_risk_desc, id_2_label_index, test_true_y,
                                   test_pred_y)

            budgets = [10, 20, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
            risk_correct = [0] * len(budgets)
            base_correct = [0] * len(budgets)
            for i in range(test_num):
                for budget in range(len(budgets)):
                    if i < budgets[budget]:
                        pair_id = id_2_VaR_risk[i][0]
                        _index = id_2_label_index.get(pair_id)
                        if test_true_y[_index] != test_pred_y[_index]:
                            risk_correct[budget] += 1
                        pair_id = id_2_risk_desc[i][0]
                        _index = id_2_label_index.get(pair_id)
                        if test_true_y[_index] != test_pred_y[_index]:
                            base_correct[budget] += 1


            risk_loss_criterion = risk_model.RiskLoss(my_risk_model)
            risk_loss_criterion = risk_loss_criterion.cuda()

            rule_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_discrete(), dtype=torch.float64).cuda()
            machine_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_continue(), dtype=torch.float64).cuda()
            rule_activate = torch.tensor(my_risk_model.test_data.get_rule_activation_matrix(),
                                         dtype=torch.float64).cuda()
            machine_activate = torch.tensor(my_risk_model.test_data.get_prob_activation_matrix(),
                                            dtype=torch.float64).cuda()
            machine_one = torch.tensor(my_risk_model.test_data.machine_label_2_one, dtype=torch.float64).cuda()
            risk_y = torch.tensor(my_risk_model.test_data.risk_labels, dtype=torch.float64).cuda()


            test_ids = my_risk_model.test_data.data_ids
            test_ids_dict = dict()
            for ids_i in range(len(test_ids)):
                test_ids[ids_i] = os.path.basename(
                    test_ids[ids_i])
                test_ids_dict[test_ids[ids_i]] = ids_i

            del my_risk_model

            data_len = len(risk_y)


            model.train()

            for batch_idx, (inputs, targets, paths) in enumerate(TestDataLoader):

                optimizer.zero_grad()

                idx = batch_idx
                if inputs.shape[0] < batch_size:
                    continue
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)

                index = []

                # we just need class_name and image_name
                paths = list(paths)
                for path_i in range(len(paths)):
                    paths[path_i] = os.path.basename(
                        paths[path_i])

                    index.append(test_ids_dict[paths[path_i]])


                test_pre_batch = test_pre[index]
                rule_mus_batch = rule_mus[index]
                machine_mus_batch = machine_mus[index]
                rule_activate_batch = rule_activate[index]
                machine_activate_batch = machine_activate[index]
                machine_one_batch = machine_one[index]



                bs, n_crops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)

                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    x4, xc = model(inputs)

                except:
                    xc = model(inputs)


                xc = xc.cuda().squeeze().view(bs, n_crops, -1).mean(1)



                out=xc
                out_2=1-out
                out_temp=torch.reshape(out,(-1,1))
                out_2=torch.reshape(out_2,(-1,1))
                out_2D=torch.cat((out_temp,out_2),1)
                risk_labels = risk_loss_criterion(test_pre_batch,
                                                  rule_mus_batch,
                                                  machine_mus_batch,
                                                  rule_activate_batch,
                                                  machine_activate_batch,
                                                  machine_one_batch,
                                                  out_2D, labels=None)

                with open('/home/ssd1/ltw/PMG/risk_lable.txt', 'a') as file:
                    file.write('%d\n'%(batch_idx))
                    out_l=np.array(out.cpu().detach().numpy())
                    risk_l=np.array(risk_labels.cpu().numpy())
                    target=np.array(targets.data.cpu().numpy())

                    targets=targets.cuda()
                    risk_labels=risk_labels.cuda()


                    file.write("risk_lab\n")
                    np.savetxt(file,risk_l,delimiter=',')
                    file.write("true_label\n")
                    np.savetxt(file,target,delimiter=',')
                    file.write("out_label\n")
                    np.savetxt(file,out_l,delimiter=',')
                    file.write('\n')

                    Loss= LSLLoss(out, risk_labels) * 1
                    Loss.backward()
                    optimizer.step()














    def test(pathImgTest, pathModel, batch_size):


        model = pathModel

        model.eval()
        model.cuda()

        y_score_n = torch.empty([0, 2], dtype=torch.float32)
        dis = torch.empty([0, 2048], dtype=torch.float32)

        chex=1
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(256))
        transformList.append(transforms.FiveCrop(224))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_test = transforms.Compose(transformList)

        with torch.no_grad():

            testset = ImageFolder(
                root=pathImgTest, transform=transform_test
            )
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            distribution_x4 = []
            paths = []
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):

                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    x4, xc = model(inputs)
                    distribution_x4.extend(x4.cuda().tolist())
                except:
                    xc = model(inputs)

                if chex == 1:
                    xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
                    x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)

                _, predicted = torch.max(xc.data, 1)

                dis = torch.cat((dis.cuda(), x4), dim=0)

                y_score.extend([_[1] for _ in softmax(xc.data.cpu(), axis=1)])
                y_pred.extend(predicted.cuda().tolist())
                y_true.extend(targets.cuda().tolist())
                paths.extend(path)



                y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                y_score_t = torch.tensor(y_score_t)
                varOutput_f = torch.tensor(varOutput_f)
                varOutput_n = torch.reshape(y_score_t, (-1, 1))
                varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)



            test_acc = 100.0 * accuracy_score(y_true, y_pred)
            test_f1 = 100.0 * f1_score(y_true, y_pred)
            test_recall = 100.0 * recall_score(y_true, y_pred)
            test_precision = 100.0 * precision_score(y_true, y_pred)
            test_auc = 100.0 * roc_auc_score(y_true, y_score)
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( test_acc, test_f1,
                                                                                  test_precision, test_recall,
                                                                                  test_auc))

            return test_acc,y_score_n,dis
    def Valtest(pathImgTest, pathModel, batch_size):


        model = pathModel

        model.eval()
        model.cuda()


        y_score_n = torch.empty([0, 2], dtype=torch.float32)


        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.RandomResizedCrop(224))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transform_test = transforms.Compose(transformList)

        chex=0

        with torch.no_grad():

            testset = ImageFolder(
                root=pathImgTest, transform=transform_test
            )
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            distribution_x4 = []
            distribution_xc = []
            paths = []
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):

                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    x4, xc = model(inputs)
                    distribution_x4.extend(x4.cuda().tolist())
                except:
                    xc = model(inputs)

                if chex == 1: xc = xc.squeeze().view(bs, n_crops, -1).mean(1)

                _, predicted = torch.max(xc.data, 1)
                y_score.extend([_[1] for _ in softmax(xc.data.cpu(), axis=1)])
                y_pred.extend(predicted.cuda().tolist())

                distribution_x4.extend(x4.cuda().tolist())
                distribution_xc.extend(xc.cuda().tolist())
                y_true.extend(targets.cuda().tolist())
                paths.extend(path)


                y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                y_score_t = torch.tensor(y_score_t)
                varOutput_f = torch.tensor(varOutput_f)
                varOutput_n = torch.reshape(y_score_t, (-1, 1))
                varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)



            test_acc = 100.0 * accuracy_score(y_true, y_pred)
            test_f1 = 100.0 * f1_score(y_true, y_pred)
            test_recall = 100.0 * recall_score(y_true, y_pred)
            test_precision = 100.0 * precision_score(y_true, y_pred)
            test_auc = 100.0 * roc_auc_score(y_true, y_score)
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( test_acc, test_f1,
                                                                                  test_precision, test_recall,
                                                                                  test_auc))

            return test_acc,y_score_n








