# SG-EDNE
This repository is to reproduce the results of our work "**Robust Dynamic Network Embedding via Ensembles**" at https://arxiv.org/pdf/2105.14557.pdf

<center>
    <img src="https://github.com/houchengbin/SG-EDNE/blob/main/data/Fig.jpg" width="800"/>
</center>

Fig. The overview of proposed method. At each timestep, multiple base learners (e.g., here are three) achieve ensembles. The base learner follows a Skip-Gram embedding approach. Between consecutive timesteps, each base learner inherits its previous base learner obeying an incremental learning paradigm. To enhance the diversity among base learners, random walk with restart using different restart probability is adopted to capture different levels of local-global topology (around the affected nodes by changed edges). Embeddings from each base learner are concatenated and rescaled before downstream tasks.

If you find this work is useful, please consider the following citation.
```
@misc{hou2021robust,
      title={Robust Dynamic Network Embedding via Ensembles}, 
      author={Chengbin Hou and Guoji Fu and Peng Yang and Shan He and Ke Tang},
      year={2021},
      eprint={2105.14557},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
```

## Install
```bash
conda create -n SG-EDNE python=3.8
source activate SG-EDNE 
pip install -r requirements.txt
```

## Usage
```bash
cd SG-EDNE
bash bash/DNC-Email.sh
```

The above command is for testing SG-EDNE on DNC-Email for five dynamic networks and 10 runs in one go. All results are automatically stored under the **bash** folder. Have fun :)

## Data
Due to the space limit, we only provide DNC-Email dataset. But you may download other datasets and preprocess them as described in the section IV-A in [our paper](https://arxiv.org/pdf/2105.14557.pdf). After that, just follow the similar approach as used in bash/DNC-Email.sh to test other datasets.

Concretely, the input dynamic network can be prepared as follows: <br>
1) Follow section IV-A in [our paper](https://arxiv.org/pdf/2105.14557.pdf) to divide the edge streams into slices. <br>
2) Use [Networkx](https://networkx.org/) to construct each snapshot based on the above slices. <br>
3) Create an empty dynamic network as a empty python list, called DynG.
4) Append each snapshot to the dynamic network DynG in the chronological order.
5) Adopt [Pickle](https://docs.python.org/3/library/pickle.html) to store DynG as a .pkl file.

Please let us know if you have any questions.
