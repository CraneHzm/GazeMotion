from torch.utils.data import Dataset
import numpy as np
import os

class mogaze_dataset(Dataset):

    def __init__(self, data_dir, subjects, input_n, output_n, actions = 'all', sample_rate=1):
        actions = self.define_actions(actions)
        self.sample_rate = sample_rate           
        self.pose_gaze_head = self.load_data(data_dir, subjects, input_n, output_n, actions)
        
    def define_actions(self, action):
        """
        Define the list of actions we are using.

        Args
        action: String with the passed action. Could be "all"
        Returns
        actions: List of strings of actions
        Raises
        ValueError if the action is not included.
        """

        actions = ["pick", "place"]

        if action in actions:
            return [action]

        if action == "all":
            return actions
        raise( ValueError, "Unrecognised action: %d" % action )
        
    def load_data(self, data_dir, subjects, input_n, output_n, actions):
        action_number = len(actions)
        seq_len = input_n + output_n
        pose_gaze_head = []
        for subj in subjects:
            path = data_dir + "/" + subj + "/"
            file_names = sorted(os.listdir(path))
            pose_xyz_file_names = {}
            gaze_file_names = {}
            head_file_names = {}
            for action_idx in np.arange(action_number):
                pose_xyz_file_names[actions[ action_idx ]] = []
                gaze_file_names[actions[ action_idx ]] = []
                head_file_names[actions[ action_idx ]] = []
                
            for name in file_names:                
                name_split = name.split('_')
                action = name_split[0]
                if action in actions:                
                    data_type = name_split[-1][:-4]
                    if(data_type == 'xyz'):
                        pose_xyz_file_names[action].append(name)
                    if(data_type == 'gaze'):
                        gaze_file_names[action].append(name)
                    if(data_type == 'head'):
                        head_file_names[action].append(name)
                        
            for action_idx in np.arange(action_number):
                action = actions[ action_idx ]
                segments_number = len(pose_xyz_file_names[action])
                print("Reading subject {}, action {}, segments number {}".format(subj, action, segments_number))
                
                for i in range(segments_number):   
                    
                    pose_xyz_data_path = path + pose_xyz_file_names[action][i]
                    pose_xyz_data = np.load(pose_xyz_data_path)

                    num_frames = pose_xyz_data.shape[0]
                    if num_frames < seq_len:
                        continue                                           
                                        
                    gaze_data_path = path + gaze_file_names[action][i]
                    gaze_data = np.load(gaze_data_path)

                    head_data_path = path + head_file_names[action][i]
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
                    #print(seq_sel.shape)
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
    data_dir = "/scratch/hu/pose_forecast/mogaze_gazemotion/"
    input_n = 10
    output_n = 30
    actions = "pick"
    train_subjects = ['p1_1']
    test_subjects = ['p7_1']
    train_sample_rate = 2
    train_dataset = mogaze_dataset(data_dir, train_subjects, input_n, output_n, actions, train_sample_rate)
    print("Training data size: {}".format(train_dataset.pose_gaze_head.shape))
    test_dataset = mogaze_dataset(data_dir, test_subjects, input_n, output_n, actions)
    print("Test data size: {}".format(test_dataset.pose_gaze_head.shape))