import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.diffraction import fresnel_diffraction
from models.Unet import net_model
from models.Res_UNet import ResUnetPlusPlus
from utils.data_utils import PhaseDataset, DataLoader


# 基础参数配置（可扩展为yaml配置文件）
CONFIG = {
    "wavelength": 632e-9,        # 波长（米）
    "pixel_size": 0.5445e-6,     # 像素尺寸（米）
    "distance": 3e-3,            # 传播距离（米）
    "image_path": "E:\\Suda\\Large scale lens\\Python\\lightfield_PhysenNet\\1.tif",  # 目标光场路径
    "image_size": (512, 512),    # 输入图像尺寸
    "num_samples": 2000,         # 训练样本数
    "batch_size": 1,            # 批次大小
    "epochs": 1,               # 训练轮数
    "lr": 1e-3,                  # 学习率
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def edge_detect(img, kernel_size=3):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, kernel_size, kernel_size).to(CONFIG["device"])
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, kernel_size, kernel_size).to(CONFIG["device"])
    edge_x = torch.abs(torch.conv2d(img, sobel_x, padding=1))
    edge_y = torch.abs(torch.conv2d(img, sobel_y, padding=1))
    return (edge_x + edge_y).detach()  # 仅计算目标边缘（避免梯度反向到边缘算子）


def main():
    # 1. 加载目标光场并预处理
    target_img = np.array(Image.open(CONFIG["image_path"]), dtype=np.float32)
    target_img = target_img[..., 0] if len(target_img.shape)==3 else target_img  # 转单通道
    target_img = target_img / np.max(target_img)
    H, W = CONFIG["image_size"]
    target_img = np.resize(target_img, (H, W))  # 调整尺寸
    target_tensor = torch.from_numpy(target_img).float().unsqueeze(0).unsqueeze(0).to(CONFIG["device"])

    a1 = math.pi / H
    b1 = math.pi / W
    x = np.linspace(1, H, H)
    y = x
    [X, Y] = np.meshgrid(x, y)
    train_phase = np.exp(1j*(a1 * X ** 2 + b1 * Y ** 2))
 

    # 2. 初始化数据集与数据加载器
    dataset = PhaseDataset(train_phase, CONFIG["num_samples"], noise_level=0.3)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # 3. 初始化模型、优化器与损失函数
    # model = net_model().to(CONFIG["device"])
    model = ResUnetPlusPlus(1).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = torch.nn.MSELoss()

    # 4. 训练循环
    loss_history = []
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0

        # for batch_idx, noisy_target in enumerate(dataloader):
        #     noisy_target = noisy_target.to(CONFIG["device"])
        #
        #     # 前向传播：相位预测 -> 衍射计算
        #     pred_phase = model(noisy_target)
        for batch_idx, train_phase in enumerate(dataloader):
            train_phase = train_phase.to(CONFIG["device"])

            # 前向传播：相位预测 -> 衍射计算
            pred_phase = model(train_phase)
            U1 = torch.complex(torch.cos(pred_phase), torch.sin(pred_phase))
            pred_intensity = fresnel_diffraction(
                U1,
                wavelength=CONFIG["wavelength"],
                L=CONFIG["pixel_size"]*H,
                distance=CONFIG["distance"],
            )
            pred_intensity = pred_intensity.cuda()
            pred_intensity = pred_intensity.unsqueeze(0).unsqueeze(0)

            # 计算损失并优化
            loss = criterion(pred_intensity, target_tensor)

            target_edge = edge_detect(target_tensor)
            pred_edge = edge_detect(pred_intensity)
            edge_loss = criterion(pred_edge * target_edge, target_edge)
            loss += 0 * edge_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_history.append(loss_value)
            print('Loss:',loss_value)

        # 保存 loss 值到 txt 文件
        with open('loss_history.txt', 'w') as f:
            for loss in loss_history:
                f.write(f"{loss}\n")
        # print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {epoch_loss/len(dataloader):.4f}")

    # 5. 保存模型与结果
    # torch.save(model.state_dict(), "phase_retrieval_model.pth")
    visualize_results(target_img, pred_phase, pred_intensity, loss_history)


def visualize_results(target_img, pred_phase, pred_intensity, loss_history):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 目标与预测光强
    axes[0, 0].imshow(target_img, cmap='gray')
    axes[0, 0].set_title("Target Intensity")

    pred_intensity_np = pred_intensity.squeeze().cpu().detach().numpy()
    axes[0, 1].imshow(pred_intensity_np, cmap='gray')
    axes[0, 1].set_title("Predicted Intensity")

    # 预测相位
    pred_phase_np = pred_phase.squeeze().cpu().detach().numpy()
    axes[1, 0].imshow(pred_phase_np, cmap='hsv')
    axes[1, 0].set_title("Predicted Phase")

    # 损失曲线
    axes[1, 1].plot(loss_history)
    axes[1, 1].set_title("Training Loss")

    plt.tight_layout()
    plt.savefig("results.png")
    plt.show()


if __name__ == "__main__":
    main()