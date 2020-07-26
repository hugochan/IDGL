# IDGL
Code & data accompanying the paper ["Deep Iterative and Adaptive Learning for Graph Neural Networks"](https://arxiv.org/abs/1912.07832)

## Get started


### Prerequisites
This code is written in python 3. You will need to install a few python packages in order to run the code.
We recommend you to use `virtualenv` to manage your python packages and environments.
Please take the following steps to create a python virtual environment.

* If you have not installed `virtualenv`, install it with ```pip install virtualenv```.
* Create a virtual environment with ```virtualenv venv```.
* Activate the virtual environment with `source venv/bin/activate`.
* Install the package requirements with `pip install -r requirements.txt`.



You also need to install the far_ho package as follows:

* git clone https://github.com/lucfra/FAR-HO.git
* cd FAR-HO
* python setup.py install



### Run the IDGL model

* Cd into the src folder
* Run the IDGL model 5 times with different random seeds and report the performance

    ```
	python main.py -config config/cora/idgl.yml --multi_run
    ```

    ```
    python main.py -config config/wine/idgl.yml --multi_run
    ```

    ```
	python main.py -config config/breast_cancer/idgl.yml --multi_run
    ```

    ```
    python main.py -config config/mrd/idgl.yml --multi_run
    ```


