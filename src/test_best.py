import os
import warnings
from time import time
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable as V

warnings.filterwarnings('ignore')

from network_new.best import best

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
BATCHSIZE_PER_CARD = 8  # 8->4


def accuracy(pred_mask, label):
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    acc = (TP + TN) / (TP + FN + TN + FP)
    recall = TP / (TP + FN)
    iou = TP / (TP + FN + FP)
    pre = TP / (TP + FP + 1e-6)
    f1 = (2 * pre * recall) / (pre + recall + 1e-6)
    return TP, FN, TN, FP, acc, recall, iou, pre, f1


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        # print("batchsize:", batchsize)
        
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0 * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)  # .transpose(2,0,1)[None]
        # 修改图片尺寸
        img = cv2.resize(img, (1024, 1024))
        # print("img:", img.shape)

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)


def test_ce_net_vessel():
    source = './dataset/Massachusetts/road_test_image/'
    gt_root = './dataset/Massachusetts/road_test_label'
    val = os.listdir(source)
    disc = 20
    solver = TTAFrame(best)
    # 加载训练模型权重
    solver.load('./weights_lyl/best_mass_bs8(2e-5).th')
    # 预测图的输出目录
    target = './imgout_lyl/best_mass_bs8(2e-5)/'
    if not os.path.exists(target):
        os.mkdir(target)
    total_TP = []
    total_FN = []
    total_TN = []
    total_FP = []
    total_acc = []
    total_recall = []
    total_iou = []
    total_pre = []
    total_f1 = []
    threshold = 2

    for i, name in enumerate(val):
        image_path = os.path.join(source, name)
        mask = solver.test_one_img_from_path(image_path)
        # print("mask:", mask.shape)
        mask[mask > threshold] = 255
        mask[mask <= threshold] = 0

        mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        ground_truth_path = os.path.join(gt_root, name.split('.')[0] + '.tif')
        ground_truth = np.array(Image.open(ground_truth_path))
        # print("ground_truth:", ground_truth.shape)
        
        # ground_truth = np.stack([ground_truth] * 3, axis=2)
        # print("ground_truth:", ground_truth.shape)
        
        # ground_truth = ground_truth[:, :, 1]
        # print("ground_truth:",ground_truth.shape)

        ground_truth = cv2.resize(ground_truth, dsize=(np.shape(mask)[1], np.shape(mask)[0]))
        # print("ground_truth:", ground_truth.shape)
        predi_mask = np.zeros(shape=np.shape(mask))
        predi_mask[mask > disc] = 1
        gt = np.zeros(shape=np.shape(ground_truth))
        gt[ground_truth > 0] = 1

        TP, FN, TN, FP, acc, recall, iou, pre, f1 = accuracy(predi_mask[:, :, 0], gt)
        total_TP.append(TP)
        total_FN.append(FN)
        total_TN.append(TN)
        total_FP.append(FP)
        total_acc.append(acc)
        total_recall.append(recall)
        total_iou.append(iou)
        total_pre.append(pre)
        total_f1.append(f1)
        print(name.split('.')[0], "TP:", TP, "FN:", FN, "TN:", TN, "FP:", FP,
              "acc:", acc, "recall:", recall, "iou:", iou, "pre:", pre, "f1:", f1)

        ground_truth = np.stack([ground_truth] * 3, axis = 2)
        # print("ground_truth:", ground_truth.shape)
        
        # 蓝色：误检，红色：漏检
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j, 0] > ground_truth[i, j, 0]:
                    mask[i, j, 0] = 255
                    mask[i, j, 1] = 0
                    mask[i, j, 2] = 0
                if mask[i, j, 0] < ground_truth[i, j, 0]:
                    mask[i, j, 0] = 0
                    mask[i, j, 1] = 0
                    mask[i, j, 2] = 255
        cv2.imwrite(target + name.split('.')[0] + '.png', mask.astype(np.uint8))

    print("acc:", np.mean(total_acc), "acc_std", np.std(total_acc))
    print("recall:", np.mean(total_recall), "acc_recall:", np.std(total_recall))
    print("iou:", np.mean(total_iou), "acc_iou:", np.std(total_iou))
    print("pre:", np.mean(total_pre), "acc_pre:", np.std(total_pre))
    print("f1:", np.mean(total_f1), "acc_f1:", np.std(total_f1))



    test_ce_net_vessel()
