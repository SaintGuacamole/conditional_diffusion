

Make sure the LEAP submodule is initialized, then

1) Install LEAP

```shell
cd LEAP
pip install -e .
cd ..
```

Install Dependencies
```shell
#TORCH
pip install torch torchvision torchaudio

#FILES
pip install safetensors
pip install pydicom
pip install h5py

#UTIL
pip install tqdm
pip install wandb
pip install lovely-tensors
pip install fire
pip install diffusers
pip install torchmetrics
pip install transformers
pip install imageio
pip install accelerate
```

3) Install diffusion
```shell

pip install -e .
```


## Dataset

The scripts to extract datasets are located in 
--scripts

KiTS: https://www.cancerimagingarchive.net/collection/c4kc-kits/

fastMRI: https://fastmri.med.nyu.edu/