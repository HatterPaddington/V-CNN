import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PhaseDataset(Dataset):
    def __init__(self, target_image, num_samples, noise_level=0.1):
        """
        相位恢复数据集
        Args:
            target_image: 目标光强分布（numpy数组，HxW）
            num_samples: 生成样本数
            noise_level: 加性噪声强度
        """
        self.target = target_image / np.max(target_image)  # 归一化
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.H, self.W = target_image.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = np.random.uniform(-self.noise_level, self.noise_level, size=(self.H, self.W))
        noisy_target = self.target + noise
        return torch.from_numpy(noisy_target).float().unsqueeze(0)  # (1, H, W)


