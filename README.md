# IDGL

Code & data accompanying the NeurIPS 2020 paper ["Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings"](https://arxiv.org/abs/2006.13009).


## Architecture

![IDGL architecture.](images/arch.png)

## Get started


### Prerequisites
This code is written in python 3. You will need to install a few python packages in order to run the code.
We recommend you to use `virtualenv` to manage your python packages and environments.
Please take the following steps to create a python virtual environment.

* If you have not installed `virtualenv`, install it with ```pip install virtualenv```.
* Create a virtual environment with ```virtualenv venv```.
* Activate the virtual environment with `source venv/bin/activate`.
* Install the package requirements with `pip install -r requirements.txt`.




### Run the IDGL & IDGL-Anch models

* Cd into the `src` folder
* Run the IDGL model and report the performance

    ```
         python main.py -config config/cora/idgl.yml
    ```

* Run the IDGL-Anch model and report the performance

    ```
         python main.py -config config/cora/idgl_anchor.yml
    ```


* Notes: 
    - You can find the output data in the `out_dir` folder specified in the config file.
    - You can add `--multi_run` in the command to run multiple times with different random seeds. Please see `config/cora/idgl.yml` for example. 
    - To run IDGL & IDGL-Anch without the iterative learning or graph regularization components, please set `max_iter` to `0` or `graph_learn_regularization` to `False` in the config file.
    - You can download the 20News data from [here](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz), and move it to the `data` folder.




## Reference

If you found this code useful, please consider citing the following paper:

Yu Chen, Lingfei Wu and Mohammed J. Zaki. **"Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings."** In *Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Dec 6-12, 2020.*


    @article{chen2020iterative,
      title={Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings},
      author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed},
      journal={Advances in Neural Information Processing Systems},
      volume={33},
      year={2020}
    }
