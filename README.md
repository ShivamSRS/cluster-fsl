
## Experiments 

### Install requirements

`pip install -r requirements.txt`
 
### Download the Datasets

* [mini-imagenet](https://github.com/renmengye/few-shot-ssl-public#miniimagenet) 
* [tiered-imagenet](https://github.com/renmengye/few-shot-ssl-public#tieredimagenet)
* [CUB](https://github.com/wyharveychen/CloserLookFewShot/tree/master/filelists/CUB)

Create a 'data' folder.
Untar the downladed file and move it into the 'data' folder.

### Reproduce the results in the paper


#### 1. Pre-training
```
python trainit.py <path-to-json-file> -sb <save-dir> -t <save-title>
eg:  python trainit.py cfgs/pretrain_miniimagenet_wrn.json -sb logs/pretrain -t pretrain_miniimagenet_wrn
```
where `<path-to-json-file>` is the directory where the parameter config file of pretaining is saved. `<save-dir>` is the directory where the all ckpts of pretraining are saved. `<save-title>` is the directory where the current ckpt of pretraining is saved. 


#### 2. Fine-tuning
```
python trainit.py [path-to-json-file] [ckpt-path] -sb <save-dir> -t <save-title>
eg:  python trainit.py cfgs/finetune_miniimagenet_wrn_1.json --ckpt logs/pretrain/pretrain_miniimagenet_wrn/checkpoint_best.pth -sb logs/finetune -t miniimagenet_wrn_1
```
where `<path-to-json-file>` is the directory where the parameter config file of finetune is saved. `<ckpt-path>` is the directory of related pretrain-ckpt-file. 



#### 3. Cluster-FSL experirments with 100 unlabeled
```
python testit.py [path-to-json-file] [ckpt-path] -sb <save-dir> -s <operation: ssl | kmeans | MFC> -t <save-title>
eg:  python testit.py cfgs/ssl_large_miniimagenet-wrn-1.json logs/finetune/miniimagenet_wrn_1/checkpoint_best.pth -sb logs/Cluster_FSL -s MFC -t Cluster_FSL_miniimagenet_wrn_1
```
where `<path-to-json-file>` is the directory where the parameter config file of Cluster-FSL-testing is saved. `<ckpt-path>` is the directory of related finetune-ckpt-file. 



**Pre-trained weights**

[google drive](https://drive.google.com/file/d/1jHd5-_KwFKAWf89A-1Al3ePwKTM-XTIa/view?usp=sharing)
#### References
[Embedding Propagation](https://github.com/ElementAI/embedding-propagation) 