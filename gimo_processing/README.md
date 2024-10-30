## Code to process the GIMO dataset


## Usage:
Step 1: Download the dataset at https://github.com/y-zheng18/GIMO.

Step 2: Set 'dataset_path' and 'dataset_processed_path' in 'gimo_preprocessing.py' and run it to process the dataset. If you meet the error "RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported", follow this link to solve it https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported.

Step 3: It is optional but highly recommended to set 'data_path' in 'dataset_visualisation.py' to visualise and get familiar with the dataset.


## Citations

```bibtex
@inproceedings{hu24gazemotion,
	title={GazeMotion: Gaze-guided Human Motion Forecasting},
	author={Hu, Zhiming and Schmitt, Syn and Haeufle, Daniel and Bulling, Andreas},
	booktitle={Proceedings of the 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems},	
	year={2024}}
	
@inproceedings{zheng2022gimo,
	title={GIMO: Gaze-informed human motion prediction in context},
	author={Zheng, Yang and Yang, Yanchao and Mo, Kaichun and Li, Jiaman and Yu, Tao and Liu, Yebin and Liu, Karen and Guibas, Leonidas J},
	booktitle={Proceedings of the 2022 European Conference on Computer Vision},
	year={2022}}
```