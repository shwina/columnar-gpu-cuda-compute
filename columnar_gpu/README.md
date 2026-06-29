# columnar_gpu

Example `srun` commands to get a node with GPU or CPU (at UF):
```
srun --partition=hpg-b200 --gpus=1 --constraint=b200 --pty bash -i
srun -t 600 --qos=avery --account=avery --cpus-per-task=1 --mem-per-cpu=4G --pty bash -i
```

Set up the environment:
```
conda create -n test_env_may15_00 -c rapidsai -c conda-forge cudf=25.12 python=3.13 'cuda-version>=12.2,<=12.9'
```
Then install coffea and awkward:
  - Navigate to your local coffea dir from Lindsey (get it via `git clone -b jitters https://github.com/scikit-hep/coffea.git`) and pip install it via `pip install -e .`.
  - Navigate to your local awkward dir and install it via the "[Installation for developers](https://github.com/scikit-hep/awkward#installation-for-developers)" instructions. 

Now you are ready to run:
```
python run_adl_queries.py
```
