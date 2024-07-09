from torch import nn
import torch
import torch.nn.functional as F


class gaze_forecasting(nn.Module):
    def __init__(self, opt):
        super(gaze_forecasting, self).__init__()        
        self.opt = opt
        self.input_n = opt.input_n
        gaze_cnn_kernel_size = opt.gaze_cnn_kernel_size
        gaze_cnn_padding = (gaze_cnn_kernel_size -1)//2
        out_channels_1 = opt.gaze_cnn_layer_1
        out_channels_2 = opt.gaze_cnn_layer_2
        out_channels_3 = opt.gaze_cnn_layer_3
        
        self.gaze_cnn = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels=out_channels_1, kernel_size=gaze_cnn_kernel_size, padding = gaze_cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_1, self.input_n], elementwise_affine=True),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=gaze_cnn_kernel_size, padding = gaze_cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_2, self.input_n], elementwise_affine=True),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=gaze_cnn_kernel_size, padding = gaze_cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_3, self.input_n], elementwise_affine=True),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels_3, out_channels=3, kernel_size=gaze_cnn_kernel_size, padding = gaze_cnn_padding, padding_mode='replicate'),
            nn.Tanh()
            )
            
    def forward(self, src, input_n=10):
        input = src.permute(0, 2, 1)
        prediction = self.gaze_cnn(input).permute(0, 2, 1)
        prediction = prediction + src[:, -1:, :].expand(-1, input_n, -1).clone()
        prediction = F.normalize(prediction, dim=2)
        
        return prediction