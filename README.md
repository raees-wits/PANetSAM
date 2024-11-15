# PANetSAM

Leveraging the SAM encoder's feature extraction and the prototypical feature learning to enable cross domain medical image segmentation from CT to MRI .


SAM model usage as implemented by segment-anything 
## TO run the code to train, test and collect baselines can be done as follows:
- You need atleast 24GB of VRAM - NVIDIA RTX3090 is preferable for training, inference can be done on smaller GPU NVIDIA GTX1060 tested.
- create a conda environment using the `environment.yml` provided in this repo, it should have the necessary packages listed.
- download the sam model check point from [official sam repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) , choose the vit_h option.
- download the medsam model checkpoints from the medsam repo `wget https://github.com/ImagingDataCommons/IDC-Tutorials/releases/download/0.2.0/medsam_vit_b.pth`
- the dataset will be sourced here: [drive zip](https://drive.google.com/file/d/1TM29YiGtNjjHHuKzKHBypFaKkJDtpBL0/view?usp=sharing)
- The project structure is important it should look like this:
```
mkdir project
cd project

# inside project the datasets must be placed - extract the testing and training sets here
# clone the repo to this project folder
git clone https://github.com/raees-wits/PANetSAM/

cd PANetSAM
# extract the sam vit_h checkpoin to this folder
```

once inside of `PANetSAM`
```
# can train the model by running
python3 trainProto.py

# can test the model by running testing
python3 testing.py
```

To run the baselines from `PANetSAM`
```
cd baselines
# extract the medsam model checkpoint here
# copy the sam vit_H model here too as it cannot excess the one in the parent dir

# to get the MEDSAM baseline
python3 medsambaseline

# to get the SAM baseline
python3 mainSAM.py
```

- output images from PANetSAM will be located here: [images on drive](https://drive.google.com/file/d/1RhWswtCEaiB58Jp03SKf5HjBATkBKYiw/view?usp=sharing)

Prototype design based on:
- [PANet](https://github.com/kaixin96/PANet)

# Dataset Acknowledgements
[CHAOS Dataset](https://chaos.grand-challenge.org/Publications/)
