FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
WORKDIR /train
COPY . .
RUN pip3 install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple
RUN wandb login （有的话加上key，没有删掉）