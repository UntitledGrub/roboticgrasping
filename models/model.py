import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin import SwinTransformerSys

# Transformer - Convolution - Transposed Convolution
class TCT(nn.Module):

    def __init__(self, input_channels=4):
        super().__init__()

        self.act = nn.ReLU(inplace=True)

        self.grasp_quality_predictor = SwinTransformerSys(in_chans=4, embed_dim=48, num_heads=[1, 2, 4, 8])

        self.angle_width_predictor = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=7, stride=1, padding=3),
                                                    nn.BatchNorm2d(32),
                                                    self.act,
                                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
                                                    nn.BatchNorm2d(64),
                                                    self.act,
                                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                                    nn.BatchNorm2d(128),
                                                    self.act,
                                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                                    nn.BatchNorm2d(128),
                                                    self.act,
                                                    nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                                                    nn.BatchNorm2d(64),
                                                    self.act,
                                                    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                                                    nn.BatchNorm2d(32),
                                                    self.act)
        
        self.angle_cos = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.angle_sin = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.opening_width = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        rx = x
        
        quality_map = torch.sigmoid(self.grasp_quality_predictor(x))

        # Operands are different shapes, quality map is N x 224 x 224 while rx is N x 4 x 224 x 224
        # torch.mul broadcasts the quality map to the appropriate size (i.e. N x 4 x 224 x 224)
        x = torch.mul(quality_map, rx)

        x = self.angle_width_predictor(x)

        angle_cos = self.angle_cos(x)
        angle_sin = self.angle_sin(x)
        opening_width = self.opening_width(x)

        return quality_map, angle_cos, angle_sin, opening_width
    
    def compute_loss(self, xc, yc):
        y_q, y_cos, y_sin, y_width = yc
        # Note: forward pass of model is called inside this function
        q_pred, cos_pred, sin_pred, width_pred = self(xc)
        
        #q_loss = F.mse_loss(q_pred, y_q)
        #cos_loss = F.mse_loss(cos_pred, y_cos)
        #sin_loss = F.mse_loss(sin_pred, y_sin)
        #width_loss = F.mse_loss(width_pred, y_width)

        q_loss = F.smooth_l1_loss(q_pred, y_q)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': q_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': q_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': q_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

# Fully convolutional version of the architecture
class FCNN(nn.Module):

    def __init__(self, input_channels=4):
        super().__init__()

        self.act = nn.ReLU(inplace=True)

        self.predictor = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=7, stride=1, padding=3),
                                                    nn.BatchNorm2d(32),
                                                    self.act,
                                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
                                                    nn.BatchNorm2d(64),
                                                    self.act,
                                                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                                    nn.BatchNorm2d(128),
                                                    self.act,
                                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                                    nn.BatchNorm2d(128),
                                                    self.act,
                                                    nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                                                    nn.BatchNorm2d(64),
                                                    self.act,
                                                    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                                                    nn.BatchNorm2d(32),
                                                    self.act)

        self.grasp_quality = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.angle_cos = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.angle_sin = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.opening_width = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        rx = x
        
        quality_map = torch.sigmoid(self.grasp_quality(self.predictor(x)))

        # Operands are different shapes, quality map is N x 224 x 224 while rx is N x 4 x 224 x 224
        # torch.mul broadcasts the quality map to the appropriate size (i.e. N x 4 x 224 x 224)
        x = torch.mul(quality_map, rx)

        x = self.predictor(x)

        angle_cos = self.angle_cos(x)
        angle_sin = self.angle_sin(x)
        opening_width = self.opening_width(x)

        return quality_map, angle_cos, angle_sin, opening_width
    
    def compute_loss(self, xc, yc):
        y_q, y_cos, y_sin, y_width = yc
        # Note: forward pass of model is called inside this function
        q_pred, cos_pred, sin_pred, width_pred = self(xc)
        
        #q_loss = F.mse_loss(q_pred, y_q)
        #cos_loss = F.mse_loss(cos_pred, y_cos)
        #sin_loss = F.mse_loss(sin_pred, y_sin)
        #width_loss = F.mse_loss(width_pred, y_width)

        q_loss = F.smooth_l1_loss(q_pred, y_q)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': q_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': q_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': q_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

if __name__ == '__main__':

    from torchsummary import summary

    input = torch.randn(1, 4, 224, 224).cuda()

    model = TCT().cuda()

    model(input)

    summary(model, (4, 224, 224))

    model = FCNN().cuda()

    model(input)

    summary(model, (4, 224, 224))
