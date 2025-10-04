import os
import re
import json
import hydra
import torch
#from utils.torch_utils import distributed as dist
import utils.setup as setup

import copy


def _main(args, device, num_gpus=1):

    # device is now passed as parameter from main function
    #assert torch.cuda.is_available()
    #device="cuda"

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)


    args.exp.model_dir=args.model_dir

    #dist.init()
    dset=setup.setup_dataset(args)
    diff_params=setup.setup_diff_parameters(args)
    network=setup.setup_network(args, device)

    # Wrap network with DataParallel for multi-GPU support
    if num_gpus > 1 and torch.cuda.is_available():
        print(f"Wrapping network with DataParallel using {num_gpus} GPUs")
        network = torch.nn.DataParallel(network)
        # Move to primary device
        network = network.to(device)
    else:
        network = network.to(device)

    optimizer=setup.setup_optimizer(args, network)

    network_tester=copy.deepcopy(network)

    tester=setup.setup_tester(args, network=network_tester, diff_params=diff_params,  device=device) #this will be used for making demos during training
    print("setting up trainer")
    trainer=setup.setup_trainer(args, dset=dset, network=network, optimizer=optimizer, diff_params=diff_params, tester=tester, device=device) #this will be used for making demos during training
    print("trainer set up")


    # Print options.
    print()
    print('Training options:')
    print()
    print(f'Output directory:        {args.model_dir}')
    print(f'Network architecture:    {args.network.callable}')
    print(f'Dataset:    {args.dset.callable}')
    print(f'Diffusion parameterization:  {args.diff_params.callable}')
    print(f'Batch size:              {args.exp.batch}')
    print()

    # Train.
    trainer.training_loop()

def select_device(device_config, num_gpus=1):
    """Select device(s) based on configuration string"""
    if device_config == "auto":
        # Auto-select: use available GPUs up to num_gpus, otherwise CPU
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                num_gpus = min(num_gpus, device_count)
                if num_gpus == 1:
                    device_id = 0
                    torch.cuda.set_device(device_id)
                    device = torch.device(f"cuda:{device_id}")
                    print(f"Auto-selected CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
                    return device, num_gpus
                else:
                    # Multi-GPU setup
                    device_ids = list(range(num_gpus))
                    device = torch.device(f"cuda:{device_ids[0]}")
                    print(f"Auto-selected {num_gpus} CUDA devices: {[torch.cuda.get_device_name(i) for i in device_ids]}")
                    return device, num_gpus
            else:
                device = torch.device("cpu")
                print("Auto-selected CPU (CUDA available but no devices found)")
                return device, 1
        else:
            device = torch.device("cpu")
            print("Auto-selected CPU (CUDA not available)")
            return device, 1

    elif device_config == "cpu":
        device = torch.device("cpu")
        print("Using CPU (forced)")
        return device, 1

    elif device_config.startswith("cuda"):
        # Handle specific CUDA device selection
        if ":" in device_config:
            device_id = int(device_config.split(":")[1])
        else:
            device_id = 0

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_id < device_count:
                if num_gpus > 1 and device_id + num_gpus <= device_count:
                    # Multi-GPU setup starting from specified device
                    device_ids = list(range(device_id, device_id + num_gpus))
                    device = torch.device(f"cuda:{device_ids[0]}")
                    print(f"Using {num_gpus} CUDA devices starting from device {device_id}: {[torch.cuda.get_device_name(i) for i in device_ids]}")
                    return device, num_gpus
                else:
                    # Single GPU
                    torch.cuda.set_device(device_id)
                    device = torch.device(f"cuda:{device_id}")
                    print(f"Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
                    return device, 1
            else:
                print(f"Warning: CUDA device {device_id} not available, using CPU")
                device = torch.device("cpu")
                return device, 1
        else:
            print("Warning: CUDA not available, using CPU")
            device = torch.device("cpu")
            return device, 1

    else:
        print(f"Warning: Unknown device config '{device_config}', using CPU")
        device = torch.device("cpu")
        return device, 1

@hydra.main(config_path="conf", config_name="conf_piano")
def main(args):
    # Select device based on configuration
    device, num_gpus = select_device(args.device, args.num_gpus)

    # Pass device and num_gpus to _main function
    _main(args, device, num_gpus)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
