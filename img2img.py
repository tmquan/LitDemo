from functools import partial

import gradio as gr
import requests
import torch
from PIL import Image
import torchvision
import lightning as L
from lightning_app.components.serve import ServeGradio


# Credit to @akhaliq for his inspiring work.
# Find his original code there: https://huggingface.co/spaces/akhaliq/AnimeGANv2/blob/main/app.py
class ImageToImage(ServeGradio):

    inputs = gr.inputs.Image(type="pil").style(full_width=True, height=256)
    outputs = gr.outputs.Image(type="pil").style(full_width=True, height=256)
    lena = "http://www.lenna.org/len_std.jpg"
    img = Image.open(requests.get(lena, stream=True).raw)
    img.save("lena.jpg")
    examples = [["lena.jpg"]]

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(self, img):
        results = self.model(img=img)
        resized = torchvision.transforms.Resize(img.size)(results)
        return resized
        # return self.model(img=img)

    def build_model(self):
        repo = "AK391/animegan2-pytorch:main"
        model = torch.hub.load(repo, "generator", device="cpu")
        face2paint = torch.hub.load(repo, "face2paint", size=256, device="cpu")
        self.ready = True
        return partial(face2paint, model=model)


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.demo = ImageToImage()

    def run(self):
        self.demo.run()

    def configure_layout(self):
        tabs = []
        if self.demo.ready:
            tabs.append({"name": "Home", "content": self.demo})
        return tabs


app = L.LightningApp(RootFlow())