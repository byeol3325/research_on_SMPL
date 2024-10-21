# SMPL to 2D Keypoints

This project demonstrates how to generate 2D keypoints from a 3D SMPL model using pre-trained SMPL or SMPL-X models. It includes visualization of the 3D body mesh and projection of 3D keypoints onto a 2D plane for human pose estimation.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Download Pretrained SMPL Models](#download-pretrained-smpl-models)
  - [Running the Code](#running-the-code)
- [Explanation of Code](#explanation-of-code)
- [References](#references)

## Introduction
SMPL (Skinned Multi-Person Linear) is a 3D body model designed for human shape representation, widely used in computer vision and animation tasks. This project leverages either SMPL or SMPL-X models to generate 3D meshes and project them onto a 2D plane to extract keypoints, which are commonly used for:
- Human pose estimation
- Animation and motion capture
- Computer vision tasks like body tracking and action recognition

The projection of 3D joints onto 2D helps simulate how a camera captures a person's skeletal structure.

## Installation

1. Clone this repository and navigate to the project directory:
    ```bash
    git clone https://github.com/your-repo-url/smpl-to-2dkeypoints.git
    cd smpl-to-2dkeypoints
    ```
2. Activate the environment using the provided `environment.yml` file (if you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).):
    ```bash
    conda env create -f environment.yml
    conda activate smpl_env
    pip install mmhuman3d --no-deps
    ```

    or use requirements.txt
   ```bash
   pip install -r requirements.txt
   pip install mmhuman3d --no-deps
   ```

## Usage

### Download Dataset

A collection of 3D human motion and pose datasets. It includes 3D body keypoints, 3D facial mesh (FLAME), 3D body mesh (SMPL), with hand (SMPL-H), and facial expression (SMPL-X) dataset.

- motion-X dataset (3D keypoints, [homwpage](https://github.com/IDEA-Research/Motion-X), [paper](https://arxiv.org/pdf/2307.00818)])
- Human3.6M (3D keypoints, [homepage](http://vision.imar.ro/human3.6m/description.php), [paper](https://ieeexplore.ieee.org/document/6682899))
- AMASS: Archive of Motion Capture As Surface Shapes  (SMPL-H+DMPL, [homepage](https://amass.is.tue.mpg.de/), [paper](http://files.is.tue.mpg.de/black/papers/amass.pdf), [code](https://github.com/nghorbani/amass))
- CMU (3D keypoints, [homepage](http://mocap.cs.cmu.edu))
- Eyes Japan (3D keypoints, [homepage](http://mocapdata.com))
- HumanEva (3D keypoints, [paper](https://files.is.tue.mpg.de/black/papers/ehumIJCV10web.pdf))
- KIT (3D keypoints, [homepage](https://motion-database.humanoids.kit.edu/), [paper](https://ieeexplore.ieee.org/document/7251476))
- SFU (3D keypoints, [homepage](http://mocap.cs.sfu.ca))
- TCD Hands (3D keypoints, [paper](https://dl.acm.org/doi/10.1145/2159616.2159630))
- TotalCapture (3D keypoints, [homepage](https://cvssp.org/data/totalcapture/), [paper](http://www.bmva.org/bmvc/2017/papers/paper014/paper014.pdf))
- Dance with Melody (3D keypoints, [paper](https://dl.acm.org/doi/10.1145/3240508.3240526), [code](https://github.com/Music-to-dance-motion-synthesis/dataset))
- Dancing to Music (3D keypoints, [paper](https://papers.nips.cc/paper/2019/hash/7ca57a9f85a19a6e4b9a248c1daca185-Abstract.html), [code](https://github.com/NVlabs/Dancing2Music))
- Music2Dance (3D keypoints, [paper](https://dl.acm.org/doi/abs/10.1145/3485664))
- EA-MUD & Youtube-Dance3D (3D keypoints, [homepage](http://zju-capg.org/deepdance.html), [paper](https://ieeexplore.ieee.org/abstract/document/9042236), [code](https://github.com/computer-animation-perception-group/DeepDance))
- AIST++ (SMPL [homepage](https://google.github.io/aistplusplus_dataset/index.html), [paper](https://arxiv.org/abs/2101.08779), [download](https://google.github.io/aistplusplus_dataset/download.html))
- Transflower (3D keypoints, [homepage](https://metagen.ai/transflower.html), [paper](https://arxiv.org/abs/2106.13871v2), [code](https://github.com/guillefix/transflower-lightning), [colab](https://colab.research.google.com/drive/1SBEJZp3TdVbgjAP9pwsTPqaefK3QuUVj))
- PhantomDance (SMPL, [homepage](https://huiye-tech.github.io/project/dancenet3d/), [paper](https://arxiv.org/abs/2103.10206), [code](https://github.com/huiye-tech/DanceNet3D))
- VOCA (head FLAME only, [homepage](https://voca.is.tue.mpg.de/), [paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/510/paper_final.pdf), [code](https://github.com/TimoBolkart/voca))
- AGORA (SMPL-X, [homepage](https://agora.is.tue.mpg.de/), [paper](https://arxiv.org/pdf/2104.14643.pdf), [code](https://github.com/pixelite1201/agora_evaluation))
- EFT (SMPL, [paper](https://arxiv.org/abs/2004.03686), [code](https://github.com/facebookresearch/eft))
- SMPly (head FLAME only, [homepage](https://europe.naverlabs.com/research/computer-vision/mannequin-benchmark/), [paper](https://arxiv.org/abs/2012.02743), [code](https://github.com/TimoBolkart/voca))



### Download Pretrained SMPL Models
To run this project, you need to download the SMPL or SMPL-X pre-trained models:

- [Download SMPL models](https://smpl.is.tue.mpg.de/index.html)
- [Download SMPL-X models](https://smpl-x.is.tue.mpg.de/)

After downloading, place the `.pkl` model files in an appropriate directory (e.g., `pretrained_model/`) and update the path in your code.

### About the Code

You can check process step by step through jupyter notebook. [Jupyter Notebook: check_SMPL_to_keypoint.ipynb](./check_SMPL_to_keypoint.ipynb)
Once the environment is set up and the pretrained models are downloaded, you can run the code to visualize the 2D projections and 3D meshes:

1. Load the SMPL model and pass the necessary parameters (like betas, pose, etc.):
   ```python
   smpl_model = SMPL(model_path='path_to_smpl_model.pkl', gender='neutral', batch_size=1)
   ```

2. Use the model to generate 3D keypoints, then project them onto a 2D plane and visualize:
   ```python
   output = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
   keypoints_3d = output['joints']
   ```

3. Project the 3D keypoints to 2D and visualize:
   ```python
   projected_keypoints_2d = project_3d_to_2d(keypoints_3d, focal_length, image_center)
   ```

### Explanation of Code

- **smpl_model**: Loads the SMPL model and provides the structure for generating 3D human body meshes.
- **project_3d_to_2d**: Projects the 3D vertices and joints onto a 2D plane using a simple pinhole camera projection method.
- **plot_keypoints_2d**: Visualizes the projected 2D keypoints on a plane.
  
The projection helps simulate a camera's view of the person, showing how 3D body keypoints are mapped to the 2D image space.

### References

- [SMPL Model](https://smpl.is.tue.mpg.de/index.html)
- [SMPL-X Model](https://smpl-x.is.tue.mpg.de/)
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d) for SMPL implementation and keypoint mapping.

