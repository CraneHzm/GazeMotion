from utils import mogaze_dataset, seed_torch
from model import gaze_forecasting
from utils.opt import options
from utils import log
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import datetime
import torch.optim as optim
import os
import math


def main(opt):
    if opt.use_gaze == 0:
        print("predict future gaze direction")
    elif opt.use_gaze == 2:
        print("predict future head direction")
    else:
        raise ValueError('use_gaze == 0 or 2')
        
    # set the random seed to ensure reproducibility
    seed_torch.seed_torch(seed=0)
    torch.set_num_threads(2)
    
    learning_rate = opt.learning_rate
    print('>>> create model')
    net_pred = gaze_forecasting.gaze_forecasting(opt=opt).to(opt.cuda_idx)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.learning_rate)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    print('>>> loading datasets')
    
    data_dir = opt.data_dir
    input_n = opt.input_n
    output_n = opt.output_n
    actions = opt.actions
    test_participant = opt.test_id
    if test_participant == 1:        
        train_subjects = ['p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p1_1', 'p1_2']
    if test_participant == 2:
        train_subjects = ['p1_1', 'p1_2', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p2_1']
    if test_participant == 4:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p4_1']
    if test_participant == 5:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p5_1']
    if test_participant == 6:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p7_1', 'p7_3']
        test_subjects = ['p6_1', 'p6_2']
    if test_participant == 7:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2']
        test_subjects = ['p7_1', 'p7_3']
    
    train_dataset = mogaze_dataset.mogaze_dataset(data_dir, train_subjects, input_n, output_n, actions, sample_rate = opt.train_sample_rate)
    train_data_size = train_dataset.pose_gaze_head.shape
    print("Training data size: {}".format(train_data_size))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_dataset = mogaze_dataset.mogaze_dataset(data_dir, test_subjects, input_n, output_n, actions)
    valid_data_size = valid_dataset.pose_gaze_head.shape
    print("Validation data size: {}".format(valid_data_size))                
    valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # training
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTraining starts at ' + local_time)
    start_time = datetime.datetime.now()
    start_epoch = 1
    
    err_best = 1000
    best_epoch = 0
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma, last_epoch=-1)
    for epo in range(start_epoch, opt.epoch + 1):
        is_best = False            
        learning_rate = exp_lr.optimizer.param_groups[0]["lr"]
        
        train_start_time = datetime.datetime.now()
        result_train = run_model(net_pred, optimizer, is_train=1, data_loader=train_loader, opt=opt)
        train_end_time = datetime.datetime.now()
        train_time = (train_end_time - train_start_time).seconds*1000
        train_batch_num = math.ceil(train_data_size[0]/opt.batch_size)
        train_time_per_batch = math.ceil(train_time/train_batch_num)
        #print('\nTraining time per batch: {} ms'.format(train_time_per_batch))
        
        exp_lr.step()
        rng_state = torch.get_rng_state()
        if epo % opt.validation_epoch == 0:
            print("\nAction: {}, test participant: {}\n".format(actions, test_participant))
            print('>>> training epoch: {:d}, lr: {:.12f}'.format(epo, learning_rate))
            print('Training data size: {}'.format(train_data_size))
            print('Average baseline error: {:.1f} degree'.format(result_train['baseline_error_average']))            
            print('Average training error: {:.1f} degree'.format(result_train['prediction_error_average']))
            
            test_start_time = datetime.datetime.now()
            result_valid = run_model(net_pred, is_train=0, data_loader=valid_loader, opt=opt)                        
            test_end_time = datetime.datetime.now()
            test_time = (test_end_time - test_start_time).seconds*1000
            test_batch_num = math.ceil(valid_data_size[0]/opt.test_batch_size)
            test_time_per_batch = math.ceil(test_time/test_batch_num)
            #print('\nTest time per batch: {} ms'.format(test_time_per_batch))
            print('Validation data size: {}'.format(valid_data_size))

            print('Average baseline error: {:.1f} degree'.format(result_valid['baseline_error_average']))                
            print('Average validation error: {:.1f} degree'.format(result_valid['prediction_error_average']))
            
            if result_valid['prediction_error_average'] < err_best:
                err_best = result_valid['prediction_error_average']
                is_best = True
                best_epoch = epo
                
            print('Best validation error: {:.1f} degree, best epoch: {}'.format(err_best, best_epoch))
            end_time = datetime.datetime.now()
            total_training_time = (end_time - start_time).seconds/60
            print('\nTotal training time: {:.1f} min'.format(total_training_time))
            local_time = time.asctime(time.localtime(time.time()))
            print('\nTraining ends at ' + local_time)
            
            # save the results
            head = np.array(['epoch', 'lr', 'training_error', 'valid_data_size', 'baseline', 'prediction'])
            training_error = format(result_train['prediction_error_average'], ".1f")
            baseline = format(result_valid['baseline_error_average'], ".1f")
            prediction = format(result_valid['prediction_error_average'], ".1f")
            
            ret_log = np.array([epo, learning_rate, training_error, valid_data_size[0], baseline, prediction])
            if opt.use_gaze == 0:
                csv_name = 'gaze_forecasting_results'
                model_name = 'gaze_forecasting_model.pt'
            if opt.use_gaze == 2:
                csv_name = 'head_forecasting_results'
                model_name = 'head_forecasting_model.pt'
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1), file_name=csv_name)
            
            # save the model
            log.save_ckpt({'epoch': epo,
                           'lr': learning_rate,
                           'err': result_valid['prediction_error_average'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                            opt=opt,
                            file_name = model_name)
                          
        torch.set_rng_state(rng_state)

        
def eval(opt):        
    print('>>> create model')
    net_pred = gaze_forecasting.gaze_forecasting(opt=opt).to(opt.cuda_idx)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    #load model
    if opt.use_gaze == 0:
        model_name = 'gaze_forecasting_model.pt'
    if opt.use_gaze == 2:
        model_name = 'head_forecasting_model.pt'
    
    model_path = os.path.join(opt.ckpt, model_name)
    print(">>> loading ckpt from '{}'".format(model_path))
    ckpt = torch.load(model_path)
    net_pred.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    
    print('>>> loading datasets')
    data_dir = opt.data_dir
    input_n = opt.input_n
    output_n = opt.output_n
    actions = opt.actions
    test_participant = opt.test_id
    if test_participant == 1:        
        train_subjects = ['p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p1_1', 'p1_2']            
    if test_participant == 2:            
        train_subjects = ['p1_1', 'p1_2', 'p4_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p2_1']
    if test_participant == 4:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p5_1', 'p6_1', 'p6_2', 'p7_1', 'p7_3']
        test_subjects = ['p4_1']
    if test_participant == 5:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p6_1', 'p6_2', 'p7_1','p7_3']
        test_subjects = ['p5_1']
    if test_participant == 6:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p7_1', 'p7_3']
        test_subjects = ['p6_1', 'p6_2']
    if test_participant == 7:            
        train_subjects = ['p1_1', 'p1_2', 'p2_1', 'p4_1', 'p5_1', 'p6_1', 'p6_2']
        test_subjects = ['p7_1', 'p7_3']
        
    test_dataset = mogaze_dataset.mogaze_dataset(data_dir, test_subjects, input_n, output_n, actions)
    test_data_size = test_dataset.pose_gaze_head.shape
    print("Test data size: {}".format(test_data_size))                
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # test
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest starts at ' + local_time)
    start_time = datetime.datetime.now()
    print("\nAction: {}, Test participant: {}\n".format(actions, test_participant))    
    result_test = run_model(net_pred, is_train=0, data_loader=test_loader, opt=opt)       
    print('Average baseline error: {:.1f} degree'.format(result_test['baseline_error_average']))                        
    print('Average validation error: {:.1f} degree'.format(result_test['prediction_error_average']))
   
    end_time = datetime.datetime.now()
    total_test_time = (end_time - start_time).seconds/60
    print('\nTotal test time: {:.1f} min'.format(total_test_time))  
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest ends at ' + local_time)

    
def acos_safe(x, eps=1e-6):
    slope = np.arccos(1-eps) / eps
    buf = torch.empty_like(x)
    good = abs(x) <= 1-eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
    return buf

    
def run_model(net_pred, optimizer=None, is_train=1, data_loader=None, opt=None):
    if is_train == 1:
        net_pred.train()
    else:
        net_pred.eval()
            
    prediction_error_average = 0
    baseline_error_average = 0
    
    n = 0
    input_n = opt.input_n
    
    for i, (data) in enumerate(data_loader):
        batch_size, seq_n, dim = data.shape
        joint_number = opt.joint_number
        # when only one sample in this batch
        if batch_size == 1 and is_train == 1:
            continue
        n += batch_size
        data = data.float().to(opt.cuda_idx)
        
        if opt.use_gaze == 0:
            input = data.clone()[:, :input_n, joint_number*3:joint_number*3+3]
            ground_truth = data.clone()[:, input_n:input_n*2, joint_number*3:joint_number*3+3]
            baseline = input[:, -1:, :].expand(-1, input_n, -1).clone()
            prediction = net_pred(input, input_n=input_n)
                                    
        if opt.use_gaze == 2:
            input = data.clone()[:, :input_n, joint_number*3+3:joint_number*3+6]
            ground_truth = data.clone()[:, input_n:input_n*2, joint_number*3+3:joint_number*3+6]
            baseline = input[:, -1:, :].expand(-1, input_n, -1).clone()
            prediction = net_pred(input, input_n=input_n)
            
        # training process
        loss = torch.mean(acos_safe(torch.sum(ground_truth*prediction, 2)))/torch.tensor(math.pi) * 180.0
                           
        if is_train == 1:            
            optimizer.zero_grad()
            loss.backward()                        
            optimizer.step()
        
        # Calculate prediction errors
        error = torch.mean(acos_safe(torch.sum(ground_truth*prediction, 2)))/torch.tensor(math.pi) * 180.0
        prediction_error_average += error.cpu().data.numpy() * batch_size
        
        baseline_error = torch.mean(acos_safe(torch.sum(ground_truth*baseline, 2)))/torch.tensor(math.pi) * 180.0
        baseline_error_average += baseline_error.cpu().data.numpy() * batch_size
                                                                       
    result = {}
    result["prediction_error_average"] = prediction_error_average / n
    result["baseline_error_average"] = baseline_error_average / n
    
    return result

    
if __name__ == '__main__':    
    option = options().parse()
    if option.is_eval == False:
        main(option)
    else:
        eval(option)