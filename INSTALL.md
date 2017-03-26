## Installation

### Anaconda in Windows

1. Install Anaconda3_python3.6 version

2. Install tensorflow by following order(!!!TensorFlow only supports version 3.5.x of Python on Windows!!!)
Start Anaconda propmt
Execute 'conda create --name tensorflow python=3.5'
Execute 'activate tensorflow'
Execute 'conda install jupyter'
Execute 'conda install scipy'
Execute 'pip install tensorflow'
Execute 'conda install spyder'
Execute 'pip install tflearn' (Can not be imported because it needs curses library which is not supported on windows)
Execute 'conda install h5py'
Execute 'conda install matplotlib'
Execute 'pip install gym'
Execute 'conda install swig'
Execute 'conda install -c conda-forge ffmpeg'
Execute 'pip install gym[box2d]'
(if failed, a. install Visual C++ 2015 Build Tools,
            b. follow kengz's comment in https://github.com/openai/gym/issues/100, change command as follow,
               pip uninstall box2d-py
               git clone https://github.com/pybox2d/pybox2d
               cd pybox2d/
               python setup.py clean
               python setup.py build
               python setup.py install)
Create shortcut on desk from envs\tensorflow\Scripts\spyder.exe

### Plain Python with Virtualenv in Unix (Mac or Linux)

1. Install Miniconda

2. Create Virtual environment (here uses `tensorflow` as the name)

``` $ conda create --name tensorflow python=3.5 ```

3. Activate the virtual environment

``` $ source activate tensorflow ```

4. Install Conda dependencies

``` 
(tensorflow) $ conda config --append channels conda-forge
(tensorflow) $ while read requirement; do conda install --yes $requirement; done < requirements.txt 
```

5. Install Tensorflow

```
(tensorflow) $ pip install tensorflow
(tensorflow) $ pip install tflearn
```

6. Install Gym

```
(tensorflow) $ pip install gym
(tensorflow) $ pip install gym[box2d]
```

> (if happened module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'

follow kengz's comment in https://github.com/openai/gym/issues/100, change command as follow,
               pip uninstall box2d-py
               git clone https://github.com/pybox2d/pybox2d
               cd pybox2d/
               python setup.py clean
               python setup.py build
               python setup.py install
               )

> TIP: If you prefer to have the over 720 open source packages included with Anaconda, and have a few minutes and the disk space required, you can download Anaconda simply by replacing the word “Miniconda” with “Anaconda” in the examples below.

Ref: 
- https://conda.io/docs/install/quick.html
- http://stackoverflow.com/a/38609653
