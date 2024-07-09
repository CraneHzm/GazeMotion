python main_mogaze_gaze_forecasting.py --data_dir /scratch/hu/pose_forecast/mogaze_gazemotion/ --ckpt ./checkpoints/mogaze_p6/ --cuda_idx cuda:5 --joint_number 21 --test_id 6 --use_gaze 0 --train_sample_rate 2 --gamma 0.9 --output_n 10 --epoch 50;

python main_mogaze_pose_forecasting.py --data_dir /scratch/hu/pose_forecast/mogaze_gazemotion/ --ckpt ./checkpoints/mogaze_p6/ --cuda_idx cuda:5 --joint_number 21 --test_id 6 --use_gaze 0 --train_sample_rate 2 --gamma 0.95 --epoch 100;

python main_mogaze_pose_forecasting.py --data_dir /scratch/hu/pose_forecast/mogaze_gazemotion/ --ckpt ./checkpoints/mogaze_p6/ --cuda_idx cuda:5 --joint_number 21 --test_id 6 --use_gaze 0 --train_sample_rate 2 --gamma 0.95 --epoch 100 --is_eval;