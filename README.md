# SG-EDNE

## install
```bash
conda create -n SG-EDNE python=3.8
source activate SG-EDNE 
pip install -r requirements.txt
```

## usage
```bash
cd SG-EDNE
bash bash/DNC-Email.sh
```

The above command is for testing SG-EDNE on DNC-Email for five dynamic networks and 10 runs in one go. All results are automatically stored under the **bash** folder. Have fun :)

Due to the space limit, we only provide DNC-Email dataset. But you may download other datasets and preprocess them as described in our paper. After that, just follow the similar approach as in used bash/DNC-Email.sh to test other datasets.

Note that: we are ready to open the source code after the peer review process.