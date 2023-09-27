'''
    Original ResNet and ResNet with bnstance Norm Layers
    Noticing that we use resnet with bottleneck block
'''
import ipdb
import torch
import torch.nn as nn
from basemodels import cusResNet152
from torchvision.models.feature_extraction import get_graph_node_names


class ResBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels * expansion, stride=stride, kernel_size=1, bias=False)
        self.adabn = nn.ModuleList([nn.BatchNorm2d(out_channels * expansion, track_running_stats=True, affine=True) for _ in range(2)])
    
    def forward(self, x, task_idx):
        x = self.conv(x)
        x = self.adabn[task_idx](x)
        
        return x
        
class BottleNeck_with_IN(nn.Module):
    '''
        Replace BN layers in BottleNeck with bns
        
        repalce bn in the residual branch with adabn too!!!
    '''
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, task_num=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bns1 = nn.ModuleList([nn.BatchNorm2d(out_channels, track_running_stats=True, affine=True) for _ in range(task_num)])
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bns2 = nn.ModuleList([nn.BatchNorm2d(out_channels, track_running_stats=True, affine=True) for _ in range(task_num)])
        self.conv3 = nn.Conv2d(out_channels, out_channels * BottleNeck_with_IN.expansion, kernel_size=1, bias=False)
        self.bns3 = nn.ModuleList([nn.BatchNorm2d(out_channels * BottleNeck_with_IN.expansion, track_running_stats=True, affine=True) for _ in range(task_num)])
        self.relu = nn.ReLU(inplace=True)
    
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck_with_IN.expansion:
            self.shortcut1 = nn.Conv2d(in_channels, out_channels * BottleNeck_with_IN.expansion, stride=stride, kernel_size=1, bias=False)
            self.shortcut2 = nn.ModuleList([nn.BatchNorm2d(out_channels * BottleNeck_with_IN.expansion, track_running_stats=True, affine=True) for _ in range(2)])
        
        # ipdb.set_trace()
    
    def forward(self, x, task_idx=0):
        x1 = self.relu(self.bns1[task_idx](self.conv1(x)))
        x1 = self.relu(self.bns2[task_idx](self.conv2(x1)))
        x1 = self.bns3[task_idx](self.conv3(x1))
        output = self.relu(self.shortcut2[task_idx](self.shortcut1(x)) + x1)
        
        return output

class BottleNeck_with_IN_new(nn.Module):
    '''
        Replace BN layers in BottleNeck with bns
    '''
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, task_num=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bns1 = nn.ModuleList([nn.BatchNorm2d(out_channels, track_running_stats=True, affine=True) for _ in range(task_num)])
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bns2 = nn.BatchNorm2d(out_channels, track_running_stats=True, affine=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels * BottleNeck_with_IN_new.expansion, kernel_size=1, bias=False)
        self.bns3 = nn.ModuleList([nn.BatchNorm2d(out_channels * BottleNeck_with_IN_new.expansion, track_running_stats=True, affine=True) for _ in range(task_num)])
        self.relu = nn.ReLU(inplace=True)
    
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck_with_IN_new.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck_with_IN_new.expansion,
                          stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck_with_IN_new.expansion)
            )
        
        # ipdb.set_trace()
    
    def forward(self, x, task_idx=0):
        x1 = self.relu(self.bns1[task_idx](self.conv1(x)))
        x1 = self.relu(self.bns2(self.conv2(x1)))
        x1 = self.bns3[task_idx](self.conv3(x1))
        output = self.relu(self.shortcut(x) + x1)
        
        return output
    

class ResBlock(nn.Module):
    def __init__(self, block, in_channels, out_channels, num_blocks, strides, task_num=2) -> None:
        super().__init__()

        self.strides = [strides] + [1] * (num_blocks - 1)
        self.layers = nn.ModuleList([])

        for s in self.strides:
            self.layers.append(
                block(in_channels, out_channels, stride=s, task_num=task_num))
            in_channels = out_channels * block.expansion

    def forward(self, x, task_idx):
        for layer in self.layers:
            x = layer(x, task_idx)

        return x


class ResNet(nn.Module):
    def __init__(self, layers=[3, 8, 36, 3], num_classes=7, select_pos='True') -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if select_pos == 'True':
            block = BottleNeck_with_IN_new
        else:
            block = BottleNeck_with_IN
    
        self.res_block1 = ResBlock(block, 64, 64, layers[0], 1, 2) # (64 -> 64) + (64 -> 64) + (64 -> 256)
        self.res_block2 = ResBlock(block, 256, 128, layers[1], 2, 2) # (256 -> 128) + (128 -> 128) + (128, 512)
        self.res_block3 = ResBlock(block, 512, 256, layers[2], 2, 2) # (512 -> 256) + (256 -> 256) + (256 -> 1024)
        self.res_block4 = ResBlock(block, 1024, 512, layers[3], 2, 2) # (1024 -> 512) + (512 -> 512) + (512 -> 2048)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x, task_idx=0):
        # ipdb.set_trace()
        x = self.conv1(x)
        x = self.res_block1(x, task_idx)
        x = self.res_block2(x, task_idx)
        x = self.res_block3(x, task_idx)
        x = self.res_block4(x, task_idx)
        x = self.avg_pool(x)
        feature = x.view(x.size(0), -1)
        output = self.fc(feature)

        return output, feature

def resnet18(num_classes=7):
    """ return a ResNet 18 object
    """
    return ResNet(layers=[2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=7):
    """ return a ResNet 34 object
    """
    return ResNet(layers=[3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=7):
    """ return a ResNet 50 object
    """
    return ResNet(layers=[3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=7):
    """ return a ResNet 101 object
    """
    return ResNet(layers=[3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=7, select_pos=True):
    """ return a ResNet 152 object
    """
    return ResNet(layers=[3, 8, 36, 3], num_classes=num_classes, select_pos=select_pos)


if __name__ == '__main__':
    net = resnet152()
    print(net)