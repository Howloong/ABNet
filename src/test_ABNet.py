import asyncio
import concurrent.futures
import itertools
import os
import warnings
from time import time
import concurrent
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable as V

import gradio as gr
import cv2
import pandas as pd

warnings.filterwarnings("ignore")

from ABNet import ABNet

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


class TTAFrame:
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(
            self.net, device_ids=range(torch.cuda.device_count())
        )

    def test_one_img_from_path(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
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


def test_ce_net_vessel(source, model):
    # 输入 source
    disc = 20
    solver = TTAFrame(ABNet)
    # 加载训练模型权重
    solver.load(model)
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

    mask = solver.test_one_img_from_path(source)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0

    mask = np.concatenate(
        [mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2
    )
    ground_truth_path = os.path.join(
        os.path.dirname(source),
        os.path.splitext(os.path.basename(source))[0] + "-mask.png",
    )
    ground_truth = np.array(Image.open(ground_truth_path))[:, :, 1]

    mask1 = cv2.resize(
        mask, dsize=(np.shape(ground_truth)[1], np.shape(ground_truth)[0])
    )
    predi_mask = np.zeros(shape=np.shape(mask1))
    predi_mask[mask1 > disc] = 1
    gt = np.zeros(shape=np.shape(ground_truth))
    gt[ground_truth > 0] = 1

    # 指标 绘制表格
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
    params = {
        "TP": TP,
        "FN": FN,
        "TN": TN,
        "FP": FP,
        "acc": acc,
        "recall": recall,
        "iou": iou,
        "pre": pre,
        "f1": f1,
    }

    ground_truth = np.array(Image.open(ground_truth_path))[:, :, :]
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
    # mask 最终结果
    print("acc:", np.mean(total_acc), "acc_std", np.std(total_acc))
    print("recall:", np.mean(total_recall), "acc_recall:", np.std(total_recall))
    print("iou:", np.mean(total_iou), "acc_iou:", np.std(total_iou))
    print("pre:", np.mean(total_pre), "acc_pre:", np.std(total_pre))
    print("f1:", np.mean(total_f1), "acc_f1:", np.std(total_f1))
    return (
        (mask / 127.5) - 1,
        # pd.DataFrame([dict(list(params.items())[:5])]).round(2),
        # pd.DataFrame([dict(list(params.items())[5:])]).round(2),
    )


async def run_models_concurrently(image_path, model_list):
    loop = asyncio.get_event_loop()

    # 创建一个线程池来并行执行模型
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            loop.run_in_executor(executor, test_ce_net_vessel, image_path, model)
            for model in model_list
        ]

        # 等待所有任务完成并收集结果
        results = await asyncio.gather(*futures)
    return tuple(itertools.chain(*results))


if __name__ == "__main__":
    img_path = "./src/img/"
    model_path = "./src/model/"
    imgs = [f for f in os.listdir(img_path) if f.endswith(".png") and "-mask" not in f]
    imgs_label_dict = {
        os.path.join(img_path, img): os.path.join(
            img_path, img.replace(".png", "-mask.png")
        )
        for img in imgs
    }
    imgs = list(imgs_label_dict.keys())
    labels = list(imgs_label_dict.values())
    models = [
        os.path.join(model_path, model)
        for model in os.listdir(model_path)
        if model.endswith(".th")
    ]
    headers = ["TP", "FN", "TN", "FP", "acc", "recall", "iou", "pre", "f1"]
    selected_image_path = imgs[0]
    images_per_row_state = gr.State(2)
    height = 400

    with gr.Blocks() as demo:
        gr.Markdown("# 我也不知道这儿写啥反正写点东西就行吧哈哈哈")

        # 第一行，创建下拉框 选择图像
        with gr.Row():
            selected_image = gr.Dropdown(
                choices=list(imgs_label_dict.keys()),
                label="Source Image",
                value=imgs[0],
                info="选择输入图像",
            )
        with gr.Row():
            with gr.Column():
                gr.Markdown("##### Input image preview.")
                selected_image_preview = gr.Image(
                    imgs[0],
                    container=False,
                    height=height,
                    min_width=0,
                )
            with gr.Column():
                gr.Markdown("##### Input label preview.")
                selected_image_label = gr.Image(
                    labels[0],
                    container=False,
                    height=height,
                    min_width=0,
                    label="Input label preview.",
                )
                selected_image.change(
                    fn=lambda image_path: imgs_label_dict.get(
                        image_path
                    ),  # 触发时调用的函数
                    inputs=selected_image,  # 函数的输入
                    outputs=selected_image_label,  # 函数的输出
                )

            def onDropDownChange(image_path):
                selected_image_path = gr.State(image_path)
                return selected_image_path

            selected_image.change(
                fn=onDropDownChange,
                inputs=selected_image,  # 函数的输入
                outputs=selected_image_preview,  # 函数的输出
            )
            image_path = gr.State(selected_image.value)
        with gr.Row():
            btn = gr.Button("提交 & 转换", size=[100, 80])

        output_list = []
        count_per_row = 2
        for i in range(0, len(models), count_per_row):
            with gr.Row():
                for j in range(0, count_per_row):
                    with gr.Column():
                        gr.Markdown(f"#### 第{i*count_per_row+j+1}个模型:{models[i]}")
                        output_list.append(
                            gr.Image(
                                type="numpy",
                                container=False,
                                height=height,
                                min_width=0,
                            )
                        )
                # with gr.Column():
                #     output_list.append(gr.DataFrame(headers=headers[:5]))
                #     output_list.append(gr.DataFrame(headers=headers[5:]))

        # 第四行- 若干模型的输出图和表格展示
        def on_button_click():
            return asyncio.run(run_models_concurrently(selected_image_path, models))

        btn.click(
            on_button_click,
            outputs=output_list,
        )

        # @gr.render(inputs=[output_images, output_tables], triggers=[btn.click])
        # def render_gallery(images, tables):
        #     # for i, image in enumerate(images):
        #     #     gallery = gr.Gallery(value=image, key=i)
        #     for image in images:
        #         gallery = gr.Gallery(value=image)

    demo.launch()

# 启动应用
