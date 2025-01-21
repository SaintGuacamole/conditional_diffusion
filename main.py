import fire
from diffusion.experiment.conditional_diffusion_harness import ConditionalDiffusionHarness

if __name__ == "__main__":
    """
    Usage example:
    
    HELP:
    
    python main.py --help
    
    TRAIN: (FROM SCRATCH)
        python main.py --project="{{project_name}}" --run="{{run_name}}" --x0_loss_weight=1.0 --output_dir="/mnt/c/outdir/" --dataset_dir="{{dataset_dir}}" train
    
    TRAIN RESUME (FROM STEP):
        python main.py --project="{{project_name}}" --run="{{run_name}}" --x0_loss_weight=1.0 --output_dir="/mnt/c/outdir/" --dataset_dir="{{dataset_dir}}" train --step={{step}}
    
    TRAIN RESUME (FROM CHECKPOINT):
        python main.py --project="{{project_name}}" --run="{{run_name}}" --x0_loss_weight=1.0 --output_dir="/mnt/c/outdir/" --dataset_dir="{{dataset_dir}}" train --checkpoint={{checkpoint_dir}}
    
    EVALUATE (FROM STEP):
        python main.py --output_dir="/mnt/c/{{output_dir_of_training_run}}/" --dataset_dir="{{dataset_dir}}" evaluate --step={{step}}
    
    EVALUATE (FROM CHECKPOINT):
        python main.py --output_dir="" --dataset_dir="{{dataset_dir}}" evaluate --checkpoint={{checkpoint_dir}} --sample_dir={{sample_dir}}
    """

    import lovely_tensors
    lovely_tensors.monkey_patch()

    fire.Fire(ConditionalDiffusionHarness)