Norm params generation (only 1 times)

```bash
python scripts/make_normalization_params.py --inputfilename '/data/multiai/data3/HyperGraph-2212.01328/singleQuarkJet_train.root'
```

Training Step
```bash
python -m  training.train --gpu-device 0 --config config/test.yaml
```

Training Step from checkpoint
```bash
python -m  training.train --gpu-device 0 --config config/test.yaml --resume checkpoints/chek_file.ckpt
```

Evaluation Step from checkpoint and save it to npz in ./workspace/npz/ you can use option : 
@click.option("--seed", help="random seed", type=int, default=None)
@click.option("--eval_train", is_flag=True, help="Evaluate on training dataset instead of test", default=False)
@click.option("--batchsize", type=int, default=None, help="Override batch size for evaluation")

```bash
python training/eval_model.py --config_path config/test.yaml --checkpoint_path checkpoints/test/detr-epochepoch=24-val_loss=1.9253.ckpt --cuda_visible_device "1"
 ```


COMAPRE PERFORMANCE AT THE PARTICLE LEVEL WITH HGP IMPLEMNTAION
HGPflow icepp implemntation is original_refiner.npz and needs model_type = "hgp" 
```bash
python scripts/compare_particle_level.py \
    --model1_file workspace/npz/test_test.npz \
    --model1_type detr \
    --model1_name "test" \
    --model2_file workspace/npz/original_refiner.npz \
    --model2_type hgp \
    --model2_name "hgp" \
    --outputdir workspace/particle/test/ \
    --detr_conf_threshold 0.75
 ```
 
COMAPRE PERFORMANCE AT THE JET LEVEL WITH HGP IMPLEMNTAION
HGPflow icepp implemntation is original_refiner.npz and needs model_name = "hgp" for DETR model yu need model_name "detr_MODELNAME"
```bash
  python scripts/compare_jet_clustering.py \
      --model1_inputfile workspace/npz/test_test.npz --model1_name "detr_test" \
      --model2_inputfile workspace/npz/original_refiner.npz --model2_name "hgp" \
      --outputdir workspace/jet_level/test/ \
      --detr_conf_threshold 0.75
```


SEE with a WEBAPP thos plot at Particle and Jet Level and compare multiple RUN : 

```bash
streamlit run web_plot/app.py
```