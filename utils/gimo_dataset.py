from torch.utils.data import Dataset
import numpy as np
import os


class gimo_dataset(Dataset):

    def __init__(self, data_dir, input_n, output_n, train_flag = 1, sample_rate=1):
        self.sample_rate = sample_rate
        if train_flag == 1:
            data_dir = data_dir + 'train/'
        if train_flag == 0:
            data_dir = data_dir + 'test/'        
                
        self.pose_gaze_head = self.load_data(data_dir, input_n, output_n)         
        
    def load_data(self, data_dir, input_n, output_n):
        seq_len = input_n + output_n
        pose_gaze_head = []        
        file_names = sorted(os.listdir(data_dir))
        pose_xyz_file_names = []
        gaze_file_names = []
        head_file_names = []
        
        for name in file_names:
            name_split = name.split('_')
            data_type = name_split[-1][:-4]
            if(data_type == 'xyz'):
                pose_xyz_file_names.append(name)
                #print("Reading data: {}".format(name))
            if(data_type == 'gaze'):
                gaze_file_names.append(name)
            if(data_type == 'head'):
                head_file_names.append(name)

            
        segments_number = len(pose_xyz_file_names)
        #print(segments_number)
        for i in range(segments_number):
            pose_xyz_data_path = data_dir + pose_xyz_file_names[i]
            pose_xyz_data = np.load(pose_xyz_data_path)
            
            num_frames = pose_xyz_data.shape[0]
            if num_frames < seq_len:
                continue
                #raise( ValueError, "sequence length {} is larger than frame number {}".format(seq_len, num_frames))
            
            gaze_data_path = data_dir + gaze_file_names[i]
            gaze_data = np.load(gaze_data_path)               
            
            head_data_path = data_dir + head_file_names[i]
            head_data = np.load(head_data_path)                
            
            pose_gaze_head_data = pose_xyz_data
            pose_gaze_head_data = np.concatenate((pose_gaze_head_data, gaze_data), axis=1)
            pose_gaze_head_data = np.concatenate((pose_gaze_head_data, head_data), axis=1)
            
            fs = np.arange(0, num_frames - seq_len + 1)
            fs_sel = fs
            for i in np.arange(seq_len - 1):
                fs_sel = np.vstack((fs_sel, fs + i + 1))
            fs_sel = fs_sel.transpose()
            #print(fs_sel)
            seq_sel = pose_gaze_head_data[fs_sel, :]
            seq_sel = seq_sel[0::self.sample_rate, :, :]
            if len(pose_gaze_head) == 0:
                pose_gaze_head = seq_sel
            else:
                pose_gaze_head = np.concatenate((pose_gaze_head, seq_sel), axis=0)
    
        return pose_gaze_head
        
  
    def __len__(self):
        return np.shape(self.pose_gaze_head)[0]

    def __getitem__(self, item):
        return self.pose_gaze_head[item]

        
if __name__ == "__main__":
    data_dir = "/scratch/hu/pose_forecast/gimo_gazemotion/"
    input_n = 10
    output_n = 30
    train_dataset = gimo_dataset(data_dir, input_n, output_n, train_flag = 1)
    print("Training data size: {}".format(train_dataset.pose_gaze_head.shape))
    test_dataset = gimo_dataset(data_dir, input_n, output_n, train_flag = 0)
    print("Test data size: {}".format(test_dataset.pose_gaze_head.shape))