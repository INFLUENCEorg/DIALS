# IALS

Source code for the paper [Influence-Augmented Local Simulators: a Scalable Solution for Fast Deep RL in Large Networked Systems](https://proceedings.mlr.press/v162/suau22a.html):

## Requirements
[Singularity](https://sylabs.io/docs/)

## Installation
```console 
git clone git@github.com:miguelsuau/recurrent_policies.git
sudo singularity build IALS.sif IALS.def
```

## Running an experiment
To run a new experiment do:

```console
cd runners
singularity run python experimentor.py with ./configs/warehouse/local_fnn_framestack.yaml
```
This will train a new policy on the local simulator. To train on the global simulator change the config file path to `./configs/warehouse/global_fnn_framestack.yaml`.
