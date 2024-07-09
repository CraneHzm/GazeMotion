from torch import nn
import torch
from model import graph_convolution_network
import utils.util as util
import torch.nn.functional as F

        
class pose_forecasting(nn.Module):
    def __init__(self, opt):
        super(pose_forecasting, self).__init__()        
        self.opt = opt        
        self.gcn_latent_features = opt.gcn_latent_features
        self.residual_gcns_num = opt.residual_gcns_num                    
        self.joint_node_n = opt.joint_number
        self.gaze_node_n = opt.gaze_node_n
        self.input_n = opt.input_n
        self.output_n = opt.output_n
        seq_len = self.input_n+self.output_n                               
        gaze_cnn_kernel_size = opt.gaze_cnn_kernel_size
        gaze_cnn_padding = (gaze_cnn_kernel_size -1)//2
        out_channels_1 = opt.gaze_cnn_layer_1
        out_channels_2 = opt.gaze_cnn_layer_2
        out_channels_3 = opt.gaze_cnn_layer_3
        
        self.gaze_cnn = nn.Sequential(
            nn.Conv1d(in_channels = 3, out_channels=out_channels_1, kernel_size=gaze_cnn_kernel_size, padding = gaze_cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_1, seq_len], elementwise_affine=True),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=gaze_cnn_kernel_size, padding = gaze_cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_2, seq_len], elementwise_affine=True),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=gaze_cnn_kernel_size, padding = gaze_cnn_padding, padding_mode='replicate'),
            nn.LayerNorm([out_channels_3, seq_len], elementwise_affine=True),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels_3, out_channels=3, kernel_size=gaze_cnn_kernel_size, padding = gaze_cnn_padding, padding_mode='replicate'),
            nn.Tanh()
            )    
            
        # forecast human pose from eye gaze and past pose
        self.pose_gcn = graph_convolution_network.graph_convolution_network(in_features=3, latent_features=self.gcn_latent_features,
                                               node_n=self.joint_node_n + self.gaze_node_n,
                                               seq_len=seq_len,
                                               p_dropout=opt.dropout,
                                               residual_gcns_num=self.residual_gcns_num)
                                               
        dct_m, idct_m = util.get_dct_matrix(seq_len)
        self.dct_m = torch.from_numpy(dct_m).float().to(self.opt.cuda_idx)
        self.idct_m = torch.from_numpy(idct_m).float().to(self.opt.cuda_idx)     
        
    def forward(self, src, input_n=10, output_n=30):        
        idx = list(range(input_n)) + [input_n -1] * output_n
        src = src[:, idx].clone()               
        gaze_input = src.clone()[:, :, self.joint_node_n*3:self.joint_node_n*3+3].permute(0, 2, 1)            
        src = torch.matmul(self.dct_m, src)
        
        pose_input = src.clone()[:, :, :self.joint_node_n*3].permute(0, 2, 1)
        bs, seq_len, features = src.shape
        
        # perform gaze forecasting/preprocessing + pose forecasting in an end-to-end way
        gaze_out = self.gaze_cnn(gaze_input)
        gaze_out = F.normalize(gaze_out, dim=1).permute(0, 2, 1)
            
        # fuse pose and eye gaze
        if self.gaze_node_n > 0:
            gaze_out = torch.matmul(self.dct_m, gaze_out).permute(0, 2, 1)
            gaze_out = gaze_out.reshape(bs, 3, 1, input_n+output_n)
            gaze_out = gaze_out.expand(-1, -1, self.gaze_node_n, -1).clone()        
            pose_input = pose_input.reshape(bs, self.joint_node_n, 3, input_n+output_n).permute(0, 2, 1, 3)        
            gcn_input = torch.cat((pose_input, gaze_out), dim=2)
        if self.gaze_node_n == 0:
            pose_input = pose_input.reshape(bs, self.joint_node_n, 3, input_n+output_n).permute(0, 2, 1, 3)
            gcn_input = pose_input
        
        # pose forecasting
        output = self.pose_gcn(gcn_input)              
        output = output.permute(0, 2, 1, 3).reshape(bs, -1, input_n+output_n).permute(0, 2, 1)
        output = torch.matmul(self.idct_m, output)        
        output = output[:, -output_n:, :self.joint_node_n*3]
        
        return output