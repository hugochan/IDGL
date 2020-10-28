# IDGL

Code & data accompanying the NeurIPS 2020 paper ["Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings"](https://arxiv.org/abs/2006.13009).


## Get started


### Prerequisites
This code is written in python 3. You will need to install a few python packages in order to run the code.
We recommend you to use `virtualenv` to manage your python packages and environments.
Please take the following steps to create a python virtual environment.

* If you have not installed `virtualenv`, install it with ```pip install virtualenv```.
* Create a virtual environment with ```virtualenv venv```.
* Activate the virtual environment with `source venv/bin/activate`.
* Install the package requirements with `pip install -r requirements.txt`.




### Run the IDGL model

* Cd into the `src` folder
* Run the IDGL model and report the performance

    ```
         python main.py -config config/cora/idgl.yml
    ```

    ```
         python main.py -config config/wine/idgl.yml
    ```

    ```
         python main.py -config config/mrd/idgl.yml
    ```

* Note: you can add `--multi_run` in the command to run multiple times with different random seeds. Please see `config/cora/idgl.yml` for example.


### Run the IDGL-anch model

* Cd into the `src` folder
* Run the IDGL-anch model and report the performance

    ```
         python main.py -config config/cora/idgl_anchor.yml
    ```

    ```
         python main.py -config config/wine/idgl_anchor.yml
    ```

    ```
         python main.py -config config/mrd/idgl_anchor.yml
    ```

* Note: you can add `--multi_run` in the command to run multiple times with different random seeds. Please see `config/cora/idgl_anchor.yml` for example.



## Reference

If you found this code useful, please consider citing the following paper:

Yu Chen, Lingfei Wu and Mohammed J. Zaki. **"Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings."** In *Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS 2020), Dec 6-12, 2020.*


    @inproceedings{chen2020iterative,
      title={Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings},
      author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed J},
      booktitle={Proceedings of the 34th Conference on Neural Information Processing Systems},
      year={2020}
    }
    
