[docs-image]: https://readthedocs.org/projects/delta-fusion/badge/?version=latest
[docs-url]: https://delta-fusion.readthedocs.io/en/latest/?badge=latest
[pytest-url]: https://github.com/rkube/delta/workflows/pytest/badge.svg

-------------------------------------------------------------------------------
[![Documentation Status][docs-image]][docs-url]
![Unit Tests][pytest-url]

# DELTA-FUSION (a**D**aptive r**E**a**L** **T**ime **A**nalysis of big fusion data)

Delta facilitates near real-time streaming analysis of big fusion data on
remote HPC resources. It consists of modular Python code that send,
receive, and process data on different machines. The picture below gives an 
overview of Delta.
For installation and documentation, see https://delta-fusion.readthedocs.io/en/latest/index.html

![Delta Architecture](https://github.com/rkube/delta/blob/master/docs/source/delta_arch_v02.png "Delta Architecture")

Cite as:
```
@inproceedings{Kube2020,
   author = {R. Kube and R.M. Churchill and Jong Youl Choi and Ruonan Wang and Scott Klasky and C. S. Chang},
   journal = {Proceedings of the 19th Python in Science Conference},
   title = {Leading magnetic fusion energy science into the big-and-fast data lane},
   url = {http://conference.scipy.org/proceedings/scipy2020/ralph_kube.html},
   year = {2020},
}
```