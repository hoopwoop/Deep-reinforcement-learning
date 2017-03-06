# Deep-reinforcement-learning
#References
1. Deep learning: http://neuralnetworksanddeeplearning.com/
2. Reinforcement learning: https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html
3. DDPG: https://arxiv.org/pdf/1509.02971.pdf
4. DPG: http://jmlr.org/proceedings/papers/v32/silver14.pdf

#Configuration
Python3.5.3, tensorflow_cpu, openai gym, win7-64bit

#Installation
1. Install Anaconda3_python3.6 version
2. Install tensorflow by following order(!!!TensorFlow only supports version 3.5.x of Python on Windows!!!)<br />
Start Anaconda propmt<br />
Execute 'conda create --name tensorflow python=3.5'<br />
Execute 'activate tensorflow'<br />
Execute 'conda install jupyter'<br />
Execute 'conda install scipy'<br />
Execute 'pip install tensorflow'<br />
Execute 'conda install spyder'<br />
Execute 'pip install tflearn' (Can not be imported because it needs curses library which is not supported on windows)<br />
Execute 'conda install h5py'<br />
Execute 'conda install matplotlib'<br />
Execute 'pip install gym'<br />
Execute 'conda install swig'<br />
Execute 'pip install gym[box2d]' (if failed, follow kengz's comment in https://github.com/openai/gym/issues/100)<br />
Create shortcut on desk from envs\tensorflow\Scripts\spyder.exe
