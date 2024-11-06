import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import gradio as gr
from torchvision.transforms import transforms
# model=models.resnet18(pretrained=True)
# model.fc=nn.Linear(model.fc.in_features,10)
t=transforms.Compose([transforms.ToTensor(),
                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                     transforms.RandomHorizontalFlip(0.5),
                     transforms.RandomRotation(10)])
class_names=['c1','c2','c3','c4','c5','c6','c7','c8','c9']
model=torch.load('best_model.pth')
def predict(image):
    image=t(image).unsqueeze(0)
    with torch.no_grad():
        output=model(image)
        _,predicted=torch.max(output,1)
        predicted_class=predicted.item()
        return predicted_class
    
#setup gradio interface
interface=gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs='text',
    title='cifar dataset prediction',
    description='upload an image to get its class predicted')

interface.launch(share=True)