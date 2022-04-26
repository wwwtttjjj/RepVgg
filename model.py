from audioop import bias
from math import fabs
from operator import mod
from os import devnull
from time import gmtime
from tkinter.messagebox import NO
from tkinter.tix import Tree
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import copy
'''
conv_bn:to fuse these two layer to one layer
'''
def conv_bn(input, output, kernel_size,stride,padding,groups = 1):
    result = nn.Sequential()
    result.add_module('conv',nn.Conv2d(in_channels=input, out_channels=output, kernel_size=kernel_size,stride=stride,padding=padding,groups=groups))
    result.add_module('bn',nn.BatchNorm2d(num_features=output))
    return result

'''
repvgg block
'''
class RepVggblock(nn.Module):
    def __init__(self, input, output, kernel_size, stride = 1, padding = 0,dilation = 1,groups = 1,padding_mode = 'zeros', deploy = False):
        super().__init__()
        self.deploy = deploy
        self.input = input
        self.output = output
        self.groups = groups

        assert(kernel_size == 3)
        assert(padding == 1)
        self.se = nn.Identity()
        self.nonlinearity = nn.ReLU()

        if deploy:
            #default
            self.combine_para = nn.Conv2d(in_channels=input, out_channels=output,kernel_size=kernel_size,stride=stride,padding=padding)
        else:
            self.rbr_idenity = nn.BatchNorm2d(num_features=input) if output == input and stride==1 else None
            self.rbr_dense = conv_bn(input=self.input,output=self.output,kernel_size=kernel_size,stride=stride,padding=padding,groups=groups)
            self.rbr_1x1 = conv_bn(input=self.input,output=self.output,kernel_size=1,stride=stride,padding=0,groups=groups)

        self._init_weight()
    def forward(self, x):
        if hasattr(self, 'combine_para'):
            return self.nonlinearity(self.se(self.combine_para(x)))

        if self.rbr_idenity == None:
            idout = 0
        else:
            idout = self.rbr_idenity(x)
        return self.nonlinearity(self.se(self.rbr_1x1(x) + self.rbr_dense(x) + idout))
    '''
    dont konw what mean of this loss
    '''
    def get_custom_l2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1,1,1,1).torch.detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1,1,1,1).torch.detach()
        l2_loss_circle = (K3**2).sum()-(K3[:,:,1:2,1:2] ** 2).sum()
        eq_kernel = K3[:,:,1:2,1:2]*t3+K1*t1

        l2_loss_eq_kernel = (eq_kernel**2/(t3**2+t1**2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_idenity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    '''
    1x1 kernel_size padding 3x3 kernel_size by zeros
    '''
    def _pad_1x1_to_3x3_tensor(self,kernel_1x1):
        if kernel_1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel_1x1,[1,1,1,1])
            # return nn.functional.pad(kernel_1x1,[1,1,1,1])
    '''
    fuse conv(x) and bn(x) to bn(conv(x))
    '''
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert(isinstance(branch, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.input // self.groups
                kernel_value = np.zeros((self.input, input_dim, 3,3), dtype=np.float32)
                for i in range(self.input):
                    kernel_value[i, i%input_dim,1,1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1,1,1,1)
        return kernel * t, beta - running_mean * gamma / std
    '''
    after finish training mode, can convert model to anonther way to inference,named structrue re-parameters.can save about 30% time.
    '''
    def switch_to_deploy(self):
        if hasattr(self, 'combine_para'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.combine_para = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                        kernel_size=self.rbr_dense.conv.kernel_size, stride = self.rbr_dense.conv.stride,
                                        padding= self.rbr_dense.conv.padding, dilation = self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups,bias = True)
        self.combine_para.weight.data = kernel
        self.combine_para.bias.data = bias
        for para in self.parameters():
            para.detach()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0,mode='fan_in',nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias,0)


                
class RepVgg(nn.Module):
    def __init__(self, num_blocks, num_classes=1000,width_multiplier = None,override_groups_map = None,deploy = False):
        super(RepVgg, self).__init__()

        assert(len(width_multiplier) == 4)
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.cur_layer_idx = 1

        self.input = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVggblock(input=3, output=self.input,kernel_size=3,stride=2,padding=1,deploy=self.deploy)
        self.stage1 = self._make_stage(planes=int(64*width_multiplier[0]), num_blcoks=num_blocks[0],stride=2)
        self.stage2 = self._make_stage(planes=int(128*width_multiplier[1]), num_blcoks=num_blocks[1],stride=2)
        self.stage3 = self._make_stage(planes=int(256*width_multiplier[2]), num_blcoks=num_blocks[2],stride=2)
        self.stage4 = self._make_stage(planes=int(512*width_multiplier[3]), num_blcoks=num_blocks[3],stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.result = nn.Linear(in_features=int(512*width_multiplier[3]), out_features=num_classes)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.result(x)
        return x

    def _make_stage(self, planes, num_blcoks, stride):
        strides = [stride] + [1] * (num_blcoks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVggblock(input=self.input,output=planes,kernel_size=3,stride=stride,padding=1,groups=cur_groups,deploy=self.deploy))
            self.input = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
def create_RepVgg_A0(num_classes = 1000,deploy = False):
    return RepVgg(num_blocks=[2,4,14,1], num_classes=num_classes,
    width_multiplier=[0.75,0.75,0.75,2.5],override_groups_map=None,deploy=deploy)

fun_dict = {
    'RepVgg-A0':create_RepVgg_A0
}
def get_RepVgg_func_by_name(name):
    return fun_dict[name]
'''
convert(training model to inference model)
'''
def repVgg_model_convert(model:torch.nn.Module, save_path = None, do_copy = True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model