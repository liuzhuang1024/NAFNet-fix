import torch
import torch.nn as nn
import torchvision.transforms.functional as Ft
from PIL import ImageFilter
from torchvision.models import vgg16, VGG16_Weights


def gram_matrix(i_input):
    a, b, c, d = i_input.size()
    features = i_input.view(a * b, c * d)
    Gm = torch.mm(features, features.t())
    return Gm.div(a * b * c * d)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.vgg = vgg16()
        self.vgg.load_state_dict(torch.load("vgg16-397923af.pth", map_location="cpu"))
        features = list(self.vgg.features)[:23]
        self.features = nn.ModuleList(features).eval()
    
    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in {3, 8, 15, 22}:
                results.append(x)  
        return results


class FFTloss(torch.nn.Module):
    def __init__(self, loss_weight=1e-3, reduction='mean'):
        super(FFTloss, self).__init__()
        self.loss_weight = loss_weight  
        self.reduction = reduction

    def forward(self, img1, img2):
        B, C, H, W = img1.shape

        # 使用FFT计算频谱
        fft_image1 = torch.fft.rfftn(img1, dim=(-3, -2, -1))
        fft_image1 = torch.stack([fft_image1.real, fft_image1.imag])
        fft_image2 = torch.fft.rfftn(img2, dim=(-3, -2, -1))
        fft_image2 = torch.stack([fft_image2.real, fft_image2.imag])
        matrix = torch.sqrt(((fft_image1-fft_image2)**2).mean(-1, keepdim=False)).mean()
        return self.loss_weight * matrix
    
class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.model = Vgg16()
        self.criterion = torch.nn.MSELoss()

        if torch.cuda.is_available():
           self.model.cuda()
           self.criterion.cuda()
    
    
    def compute_color_loss(self, i_input, target):
        num_img = i_input.size()[0]
        color_loss = 0
        for i in range(num_img):
            input_blur = Ft.gaussian_blur(i_input[i, :, :, :], 11)
            target_blur = Ft.gaussian_blur(target[i, :, :, :], 11)
            color_loss += self.criterion(input_blur, target_blur)
        return color_loss/num_img
    
    def compute_content_loss(self, input_feats, target_feats):
        nr_feats = len(input_feats)
        content_loss = 0
        for i in range(nr_feats):
            content_loss += self.criterion(input_feats[i], target_feats[i]).item()
        return content_loss/nr_feats

    def compute_style_loss(self, input_feats, target_feats):
        nr_feats = len(input_feats)
        style_loss = 0
        for i in range(nr_feats):
            gi = gram_matrix(input_feats[i])
            gt = gram_matrix(target_feats[i])
            style_loss += self.criterion(gt, gi).item()
        return style_loss/nr_feats

    def compute_perceptual_loss(self, synthetic, real):
        input_feats = self.model(synthetic)
        target_feats = self.model(real)
        
        color_loss = self.compute_color_loss(synthetic, real)
        style_loss = self.compute_style_loss(input_feats, target_feats)
        content_loss = self.compute_content_loss(input_feats, target_feats)
        return color_loss, style_loss, content_loss


    def forward(self, pred, target):
        return self.loss_weight * sum([weight * loss for weight, loss in zip([1.0, 1e-3, 1e-1], self.compute_perceptual_loss(pred, target))])