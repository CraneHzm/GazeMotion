# GazeMotion: Gaze-guided Human Motion Forecasting
Project homepage: https://zhiminghu.net/hu24_gazemotion.


## Abstract
```
We present GazeMotion – a novel method for human motion forecasting that combines information on past human poses with human eye gaze.
Inspired by evidence from behavioural sciences showing that human eye and body movements are closely coordinated, GazeMotion first predicts future eye gaze from past gaze, then fuses predicted future gaze and past poses into a gaze-pose graph, and finally uses a residual graph convolutional network to forecast body motion. 
We extensively evaluate our method on the MoGaze, ADT, and GIMO benchmark datasets and show that it outperforms state-of-the-art methods by up to 7.4% improvement in mean per joint position error.
Using head direction as a proxy to gaze, our method still achieves an average improvement of 5.5%.
We finally report an online user study showing that our method also outperforms prior methods in terms of perceived realism.
These results show the significant information content available in eye gaze for human motion forecasting as well as the effectiveness of our method in exploiting this information.
```


## Environments:
Ubuntu 22.04
python 3.8+
pytorch 1.8.1
cudatoolkit 11.1


## Usage:
Step 1: Create the environment
```
conda env create -f ./environments/gazemotion.yaml -n gazemotion
conda activate gazemotion
```


Step 2: Follow the instructions in './adt_processing/', './gimo_processing/', and './mogaze_processing/' to process the datasets.


Step 3: Set 'data_dir' and 'cuda_idx' in 'train_mogaze_xx.sh' (xx for p1, p2, p4, p5, p6, or p7) to evaluate on different participants. By default, 'train_mogaze_xx.sh' first trains the model from scratch and then tests the model. If you only want to evaluate the pre-trained models, please comment the training commands (the commands without the 'is_eval' setting).


Step 4: Set 'data_dir' and 'cuda_idx' in 'train_adt.sh' to evaluate. By default, 'train_adt.sh' first trains the model from scratch and then tests the model. If you only want to evaluate the pre-trained models, please comment the training commands (the commands without the 'is_eval' setting).


Step 5: Set 'data_dir' and 'cuda_idx' in 'train_gimo.sh' to evaluate. By default, 'train_gimo.sh' first trains the model from scratch and then tests the model. If you only want to evaluate the pre-trained models, please comment the training commands (the commands without the 'is_eval' setting).


## Citation

```bibtex
@inproceedings{hu24_gazemotion,
	title={GazeMotion: Gaze-guided Human Motion Forecasting},
	author={Hu, Zhiming and Schmitt, Syn and Haeufle, Daniel and Bulling, Andreas},
	booktitle={Proceedings of the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems},	
	year={2024}}
```