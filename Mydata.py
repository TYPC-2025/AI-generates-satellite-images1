import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from models import DiffuseNetWithForgeryDetection
from PIL import Image
import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


class ForgeryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        for label in ['authentic', 'fake']:
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                raise FileNotFoundError(f"标签目录不存在: {label_dir}")

            for city in os.listdir(label_dir):
                city_dir = os.path.join(label_dir, city)
                if os.path.isdir(city_dir):
                    files = [f for f in os.listdir(city_dir) if f.endswith(('.jpg', '.png'))]
                    if not files:
                        print(f"警告: {city_dir} 目录中没有找到图像文件")
                    for filename in files:
                        self.image_paths.append(os.path.join(city_dir, filename))
                        self.labels.append(0 if label == 'authentic' else 1)

        if not self.image_paths:
            raise ValueError("没有找到任何有效的图像文件")
        print(f"加载了 {len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    # 设备检测与配置
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.mem_get_info()[1] / 1024 ** 3:.2f}GB")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("CUDA不可用原因：")
        print("- PyTorch未安装CUDA版本" if not torch.cuda.is_available() else "")
        print("- 显卡驱动未正确安装")

    try:
        # 增强数据预处理
        train_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # 数据集准备
        data_dir = r'E:\仓库\Github仓库\AI检测卫星图像\data'
        full_dataset = ForgeryDataset(data_dir)

        # 划分训练集和验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # 应用不同预处理
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        # 样本均衡处理
        class_counts = np.bincount([full_dataset.labels[i] for i in train_dataset.indices])
        class_weights = 1. / class_counts
        sample_weights = class_weights[[full_dataset.labels[i] for i in train_dataset.indices]]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        # 数据加载器配置
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # 模型初始化
        model = DiffuseNetWithForgeryDetection().to(device)
        if device.type == 'cuda':
            model = model.half()

        # 优化器配置
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )

        # 训练循环
        best_acc = 0.0
        for epoch in range(30):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()

                if device.type == 'cuda':
                    images = images.half()

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    detection, _ = model(images)
                    loss = criterion(detection.squeeze(), labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch: {epoch + 1}/{30} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

            # 验证阶段
            model.eval()
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.cpu().numpy()

                    if device.type == 'cuda':
                        images = images.half()

                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        detection, _ = model(images)

                    probs = torch.sigmoid(detection).cpu().numpy().flatten()
                    all_probs.extend(probs)
                    all_labels.extend(labels)

            # 计算评估指标
            val_auc = roc_auc_score(all_labels, all_probs)
            val_f1 = f1_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
            val_acc = (np.array(all_probs) > 0.5).astype(int) == np.array(all_labels)
            val_acc = val_acc.mean()

            scheduler.step(val_auc)

            print(f"\nEpoch {epoch + 1} 验证结果:")
            print(f"准确率: {val_acc:.2%} | AUC: {val_auc:.4f} | F1 Score: {val_f1:.4f}")

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print("发现新的最佳模型，已保存！")

        # 最终测试
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()

        # 测试代码（可根据需要添加完整测试流程）

    except Exception as e:
        print(f"程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()
