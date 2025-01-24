import sys
import os

from tqdm import tqdm

sys.path.append(f"{os.path.dirname(__file__)}/net/deep/AAMIFNet")
sys.path.append(f"{os.path.dirname(__file__)}/net/deep/AMIFNet")
sys.path.append(f"{os.path.dirname(__file__)}/net/mass/AAMIFNet")
sys.path.append(f"{os.path.dirname(__file__)}/net/mass/ADEFNet")


import warnings
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from TTAFrame import TTAFrame
from net.deep.AABNet import AABNet as deep_AABNet
from net.deep.AADEFNet import AADEFNet as deep_AADEFNet
from net.deep.AAMIFNet.AAMIFNet import AAMIFNet as deep_AAMIFNet
from net.deep.ABNet import ABNet as deep_ABNet
from net.deep.ADEFNet import ADEFNet as deep_ADEFNet
from net.deep.AMIFNet.AMIFNet import AMIFNet as deep_AMIFNet
from net.deep.Deeplabv3_plus import Deeplabv3_plus as deep_Deeplabv3_plus
from net.deep.UNet import UNet as deep_UNet
from net.mass.AABNet import AABNet as mass_AABNet
from net.mass.AADEFNet import AADEFNet as mass_AADEFNet
from net.mass.AAMIFNet.AAMIFNet import AAMIFNet as mass_AAMIFNet
from net.mass.ABNet import ABNet as mass_ABNet
from net.mass.ADEFNet.ADEFNet import ADEFNet as mass_ADEFNet
from net.mass.AMIFNet import AMIFNet as mass_AMIFNet
from net.mass.Deeplabv3_plus import Deeplabv3_plus as mass_Deeplabv3_plus
from net.mass.UNet import UNet as mass_UNet

warnings.filterwarnings("ignore")

BATCHSIZE_PER_CARD = 8

MASSACHUSETTS = "Massachusetts"
DEEPGLOBE = "deepGlobe"


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


def test_ce_net_vessel(img, model, weight):
    # 输入 source
    disc = 20
    solver = TTAFrame(model)
    # 加载训练模型权重
    solver.load(weight)
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

    mask = solver.test_one_img_from_path(img)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0

    mask = np.concatenate(
        [mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2
    )
    ground_truth_path = os.path.join(
        os.path.dirname(img),
        os.path.splitext(os.path.basename(img))[0] + "-mask.png",
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
        "模型名": os.path.splitext(os.path.basename(weight))[0],
        "TP": TP,
        "FN": FN,
        "TN": TN,
        "FP": FP,
        "acc": acc,
        "recall": recall,
        "pre": pre,
        "iou": iou,
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
    return [(mask / 127.5) - 1, {weight: params}]


def run_models_inorder(image_path, net_list, weight_list):
    all_params = {}
    all_componnents_source = []
    idx = 1
    for net, weight in tqdm(zip(net_list, weight_list)):
        image, param = test_ce_net_vessel(image_path, net, weight)
        idx += 1
        all_componnents_source.append(image)
        all_params.update(param)
    all_componnents_source.append(
        pd.DataFrame().from_dict(all_params, orient="index").round(2)
    )
    return tuple(all_componnents_source)


if __name__ == "__main__":
    dataset_DeepGlobe = {
        "path": f"{os.path.dirname(__file__)}/img/DeepGlobe",
        "type": DEEPGLOBE,
        "name": "DeepGlobe",
    }
    dataset_CHN = {
        "path": f"{os.path.dirname(__file__)}/img/CHN6-CUG",
        "type": DEEPGLOBE,
        "name": "CHN6-CUG",
    }
    net_deep_list = [
        deep_UNet,
        deep_Deeplabv3_plus,
        deep_ABNet,
        deep_AABNet,
        deep_AMIFNet,
        deep_ADEFNet,
        deep_AAMIFNet,
        deep_AADEFNet,
    ]

    dataset_Massachusetts = {
        "path": f"{os.path.dirname(__file__)}/img/Massachusetts",
        "type": MASSACHUSETTS,
        "name": "Massachusetts",
    }
    dataset_SpaceNet = {
        "path": f"{os.path.dirname(__file__)}/img/SpaceNet",
        "type": MASSACHUSETTS,
        "name": "SpaceNet",
    }
    net_mass_list = [
        mass_UNet,
        mass_Deeplabv3_plus,
        mass_ABNet,
        mass_AABNet,
        mass_AMIFNet,
        mass_ADEFNet,
        mass_AAMIFNet,
        mass_AADEFNet,
    ]

    datasets = [dataset_DeepGlobe, dataset_CHN, dataset_Massachusetts, dataset_SpaceNet]
    headers = ["模型名", "TP", "FN", "TN", "FP", "Acc", "Rec", "Pre", "IoU", "F1"]

    height = 150

    with gr.Blocks(
        css="""
                    footer {
                        visibility: hidden
                    }

                    #table table {
                        overflow-x: hidden !important;
                        overflow-y: hidden !important;
                    }

                    #table span {
                        font-family: 'Microsoft YaHei', '微软雅黑', sans-serif !important;
                        text-align: center;
                    }

                    .sort-button {
                        display: none !important;
                    }
                    .image img{
                        position: absolute!important;
                        top: 13px !important;
                    }
                   """
    ) as demo:
        title = """
                <center> 
                <h1> 道路提取系统 </h1>
                </center>
                """
        with gr.Row():
            gr.HTML(title)

        # 第 1 行，创建下拉框
        with gr.Row():
            # 选择数据集
            selected_dataset = gr.Dropdown(
                choices=[
                    (dataset.get("name"), i) for i, dataset in enumerate(datasets)
                ],
                value=0,
                type="value",
                show_label=False,
                info="选择数据集",
            )

            # 拼接两个列表
            default_image_list = []
            for dirpath, dirnames, filenames in os.walk(datasets[0].get("path")):
                for filename in filenames:
                    if "-mask" not in filename and filename.endswith(".png"):
                        default_image_list += [os.path.join(dirpath, filename)]
            for dirpath, dirnames, filenames in os.walk(datasets[1].get("path")):
                for filename in filenames:
                    if "-mask" not in filename and filename.endswith(".png"):
                        default_image_list += [os.path.join(dirpath, filename)]
            selected_image = gr.Dropdown(
                choices=[(Path(image).stem, image) for image in default_image_list],
                value=default_image_list[0],
                show_label=False,
                info="选择输入图像",
            )

            def selected_dataset_change(idx: tuple):
                lst = []
                for dirpath, dirnames, filenames in os.walk(datasets[idx].get("path")):
                    for filename in filenames:
                        if "-mask" not in filename and filename.endswith(".png"):
                            lst += [os.path.join(dirpath, filename)]
                return gr.update(
                    choices=[(Path(image).stem, image) for image in lst],
                    value=lst[0],
                    show_label=False,
                    info="选择输入图像",
                )

            selected_dataset.change(
                fn=selected_dataset_change,
                inputs=selected_dataset,
                outputs=selected_image,
            )
        # 第 2 行，创建预览原图像和label
        with gr.Row():
            with gr.Column():
                gr.Markdown("##### 高分辨率遥感影像")
                selected_image_preview = gr.Image(
                    default_image_list[0],
                    container=False,
                    height=height,
                )
            with gr.Column():
                gr.Markdown("##### 对应道路标签")
                selected_image_label = gr.Image(
                    default_image_list[0].replace(".png", "-mask.png"),
                    container=False,
                    height=height,
                )

            def onDropDownChange(image_path: str):
                return image_path, image_path.replace(".png", "-mask.png")

            selected_image.change(
                fn=onDropDownChange,
                inputs=selected_image,  # 函数的输入
                outputs=[selected_image_preview, selected_image_label],  # 函数的输出
            )
            image_path = gr.State(selected_image.value)
        with gr.Row():
            btn = gr.Button("提交 & 转换")

        output_list = []
        count_per_row = 4
        idx = 0
        for i in range(0, 8, count_per_row):
            # with gr.Group():
            with gr.Row(equal_height=True):
                for j in range(0, count_per_row):
                    output_list.append(
                        gr.Image(
                            type="numpy",
                            container=True,
                            width=height,
                            height=height + 40,
                            min_width=height,
                            label=net_deep_list[idx]().get_name(),
                            show_label=True,
                            interactive=False,
                            elem_classes=["image"],
                        )
                    )
                    idx += 1
        output_list.append(gr.DataFrame(headers=headers, elem_id="table"))

        # 第四行- 若干模型的输出图和表格展示
        def on_button_click(selected_dataset: int, selected_image: str):
            if 0 <= selected_dataset <= 1:
                model_list = net_deep_list
                weight_list = [
                    f"{os.path.dirname(__file__)}/weight/deep/{model().get_name()}.th" for model in model_list
                ]
            else:
                model_list = net_mass_list
                weight_list = [
                    f"{os.path.dirname(__file__)}/weight/mass/{model().get_name()}.th" for model in model_list
                ]
            return run_models_inorder(selected_image, model_list, weight_list)

        btn.click(
            on_button_click,
            inputs=[selected_dataset, selected_image],
            outputs=output_list,
        )

        demo.launch()
