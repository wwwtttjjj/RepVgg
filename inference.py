import torch
import torchvision
from PIL import Image
import argparse

translate = ["dog", "horse", "elephant", "butterfly",  "chicken", "cat",  "cow", "sheep",  "Spider","squirrel"]
parse = argparse.ArgumentParser()
parse.add_argument('--image-path', default='./demo/test.jpeg', help='the image path of inference image')
args = parse.parse_args()

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485, 0.456,0.406],std = [0.229, 0.224, 0.225])
])

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
modelrepVgg = torch.load("repVgg_Convert.pt", map_location=device)

if __name__ == '__main__':
    img = Image.open(args.image_path)
    img = transform(img).to(device)[None,:,:,:]
    with torch.no_grad():
        y_p = modelrepVgg(img)
        y_p = torch.argmax(y_p)
    print(translate[y_p.item()])
    