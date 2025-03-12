import tkinter as tk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import numpy as np
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 512*4*4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        x = self.main(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load("generator.pth", map_location=device))
netG.eval()
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generador de Imágenes DCGAN")
        self.resizable(0, 0)
        self.img_label = tk.Label(self)
        self.img_label.pack(padx=10, pady=10)
        self.btn_frame = tk.Frame(self)
        self.btn_frame.pack(fill="x", padx=10, pady=10)
        self.btn_generate = tk.Button(self.btn_frame, text="Generar Imagen", command=self.generate)
        self.btn_generate.pack(side="left", expand=True, fill="x", padx=5)
        self.btn_save = tk.Button(self.btn_frame, text="Guardar Imagen", command=self.save)
        self.btn_save.pack(side="right", expand=True, fill="x", padx=5)
        self.generated_image = None
    def generate(self):
        noise = torch.randn(1, 100, device=device)
        with torch.no_grad():
            fake = netG(noise)
        img = fake.squeeze(0).cpu().numpy()
        img = (img + 1) / 2
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype("uint8")
        self.generated_image = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(self.generated_image.resize((256, 256)))
        self.img_label.config(image=self.photo)
    def save(self):
        if self.generated_image is not None:
            self.generated_image.save("generated_image.png")
app = App()
app.mainloop()
