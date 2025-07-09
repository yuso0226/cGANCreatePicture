import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

# ハイパーパラメータ
latent_dim = 10
n_classes = 10
img_size = 28

# 使用デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator の定義
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# モデルのロード（weights_only=False を明示）
generator = torch.load("GANgenerator.pth", map_location=device, weights_only=False)
generator.to(device)
generator.eval()

# Streamlit UI
st.title("🎨 条件付きGANによる手書き数字生成")
st.markdown("指定した数字の画像を生成します。")

label = st.selectbox("生成する数字ラベル (0〜9)", list(range(10)))
n_images = st.slider("生成する画像枚数", min_value=1, max_value=10, value=5)

if st.button("画像を生成"):
    z = torch.randn(n_images, latent_dim, device=device)
    labels = torch.tensor([label] * n_images, dtype=torch.long, device=device)

    with torch.no_grad():
        gen_imgs = generator(z, labels)

    gen_imgs = (gen_imgs + 1) / 2  # [-1,1] → [0,1]

    fig, axs = plt.subplots(1, n_images, figsize=(2 * n_images, 2))
    if n_images == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        img = gen_imgs[i].cpu().squeeze()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Label {label}")
    st.pyplot(fig)
