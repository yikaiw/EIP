# EIP: Elastic Interaction of Particles
Code release for paper [**"Elastic Tactile Simulation Towards Tactile-Visual Perception"**](https://arxiv.org/pdf/2108.05013.pdf), in ACMMM (Oral) 2021

By Yikai Wang, Wenbing Huang, Bin Fang, Fuchun Sun, Chang Li.

<p align="center"><img src="assets/intro.png" width="820" /></p>


If you find our work useful for your research, please consider citing the following paper.
```
@inproceedings{wang2021eip,
  title={Elastic Tactile Simulation Towards Tactile-Visual Perception},
  author={Wang, Yikai and Huang, Wenbing and Fang, Bin and Sun, Fuchun and Li, Chang},
  booktitle={ACM International Conference on Multimedia (ACM MM)},
  year={2021}
}
```


## Dependencies
```
python==3.7.6
taichi==0.6.29
pytorch==1.6.0
pytorch3d==0.2.5
imageio==2.9.0
open3d==0.11.0
opencv-python==4.4.0
openexr==2.4.1
```


## Scripts


First, transform mesh to voxels
```
python mesh2voxel.py --path obj_path  # e.g., obj/torus.obj
```
Then perform EIP, for example,
```
python show_torus_vertical.py --name exp_name
```

Will obtain fine-grained tacile patterns like,

<p align="center"><img src="assets/patterns.png" width="920" /></p>

Simulated particles of the tactile sensor will be automatically saved, which can be viewed with [Meshlab](https://www.meshlab.net/). We adopt [Mitsuba](http://www.mitsuba-renderer.org/) for rendering.

EIP is based on [Taichi](https://github.com/taichi-dev/taichi). Thanks for the great work.

## License

EIP is released under MIT License.
