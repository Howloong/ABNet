import os
import warnings

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from TTAFrame import TTAFrame

warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
BATCHSIZE_PER_CARD = 8  # 8->4

if __name__ == "__main__":
    # 数据集image路径
    dataset_DeepGlobe_path = "./img/DeepGlobe"
    dataset_CHN6_path = "./img/CHN6-CUG"
    dataset_Massachusetts_path = "./img/Massachusetts"
    dataset_SpaceNet_path = "./img/SpaceNet"
    # 权重路径
    weight_deep_path = "./weight/deep"
    weight_mass_path = "./weight/mass"

    # 网络
    net_deep_list=[ABNet]
    imgs = [f for f in os.listdir(img_path) if f.endswith(".png") and "-mask" not in f]
    imgs_label_dict = {img: img.replace(".png", "-mask.png") for img in imgs}
    imgs = list(imgs_label_dict.keys())
    labels = list(imgs_label_dict.values())
    models = [
        "deep_bs4(2e-5).th",
        "mass_bs4(2e-5).th",
        "deep_bs4(2e-5).th",
        "mass_bs4(2e-5).th",
        "deep_bs4(2e-5).th",
        "mass_bs4(2e-5).th",
        "deep_bs4(2e-5).th",
        "mass_bs4(2e-5).th",
    ]
    # models = [os.path.join(model_path, model) for model in models]
    headers = ["模型名", "TP", "FN", "TN", "FP", "acc", "recall", "iou", "pre", "f1"]
    selected_image_path = os.path.join(img_path, imgs[0])
    images_per_row_state = gr.State(2)
    height = 150

    with gr.Blocks(css="""
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
                   """) as demo:
        title = """
                <center> 
                <h1> 道路提取系统 </h1>
                </center>
                """
        with gr.Row():
            gr.HTML(title)

        # 第一行，创建下拉框 选择图像
        with gr.Row():
            selected_dataset = gr.Dropdown(
                choices=list(imgs_label_dict.keys()),
                value=imgs[0],
                show_label=False,
                info="选择数据集",
            )
            selected_image = gr.Dropdown(
                choices=list(imgs_label_dict.keys()),
                value=imgs[0],
                show_label=False,
                info="选择输入图像",
            )
        with gr.Row():
            with gr.Column():
                gr.Markdown("##### 高分辨率遥感影像")
                selected_image_preview = gr.Image(
                    os.path.join(img_path, imgs[0]),
                    container=False,
                    height=height,
                )
            with gr.Column():
                gr.Markdown("##### 对应道路标签")
                selected_image_label = gr.Image(
                    os.path.join(img_path, labels[0]),
                    container=False,
                    height=height,
                )
                selected_image.change(
                    fn=lambda image_path: os.path.join(
                        img_path, imgs_label_dict.get(image_path)
                    ),  # 触发时调用的函数
                    inputs=selected_image,  # 函数的输入
                    outputs=selected_image_label,  # 函数的输出
                )


            def onDropDownChange(image_name):
                selected_image_path = gr.State(os.path.join(img_path, image_name))
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
        count_per_row = 4
        idx = 0
        for i in range(0, 8, count_per_row):
            # with gr.Group():
            with gr.Row(equal_height=True):
                for j in range(0, count_per_row):
                    # with gr.Column():
                    #     gr.Markdown(f"{models[i]}")
                    output_list.append(
                        gr.Image(
                            type="numpy",
                            container=True,
                            width=height,
                            height=height + 40,
                            min_width=height,
                            label=models[idx],
                            show_label=True,
                            interactive=False,
                            elem_classes=["image"]
                        )
                    )
            idx += 1
        output_list.append(gr.DataFrame(headers=headers, elem_id="table"))


        # 第四行- 若干模型的输出图和表格展示
        def on_button_click():
            return run_models_inorder(
                selected_image_path,
                # [os.path.join(model_path, model) for model in models],
                [os.path.join(weight_path, model) for model in models],
            )


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


def test_ce_net_vessel(model, img, weight):
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
    return [(mask / 127.5) - 1, {weight: params}]


def run_models_inorder(image_path, model_list):
    all_params = {}
    all_componnents_source = []
    idx = 1
    for model in model_list:
        # image, param = test_ce_net_vessel(image_path, model)
        image = Image.open(image_path)
        param = {
            f"test{idx}": {
                "模型名": f"test{idx}",
                "TP": 1.22,
                "FN": 2.22,
                "TN": 3.22,
                "FP": 4.33,
                "acc": 511.22,
                "recall": 6.32,
                "iou": 7999.22,
                "pre": 8.11,
                "f1": 12341.22,
            }
        }
        idx += 1
        all_componnents_source.append(image)
        all_params.update(param)
    all_componnents_source.append(
        pd.DataFrame()
        .from_dict(all_params, orient="index")
        .round(2)
    )
    return tuple(all_componnents_source)
