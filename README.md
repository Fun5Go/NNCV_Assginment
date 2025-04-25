# Required libraries and installation instructions.
Step 1: Download the Prerequisites
```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```
Step 2: Install other libraries 
```bash
pip install requirements.txt
```
This is in case for lack of some libraries. If it is not working, put this 'install' code in main.py cause it is running in the containner.
```bash
wandb login
pip install requirements.txt
python3 train.py \ ...
                    ...
```

## WandB
Just like the NNCV instruction.
1. Open the `.env` file using a text editor:

   ```bash
    nano .env
   ```

   If you are using MobaXTerm, you can also just open it using the file explorer windown on the right side.

2. Update the following variables:

   - `WANDB_API_KEY`: Your Weights & Biases API key (for logging experiments).
   - `WANDB_DIR`: Path to the directory where the logs will be stored.

3. Save and exit the file.

>If not use the HPC, run the code to log in WandB:
   ```bash
   wandb login
   ```
   then paste your API key (can be found in WandB personal profile)

# Steps to run the code
- If you are using the HPC to run the codes, also like the steps mentioned in the NNCV instruction:

Submit the job with the following command:

```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```
- If run it on the local computer, run the code in the terminal:
 ```bash
python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "FinalModel_Fan" \
```

#  Codalab username and TU/e email address
- Username: GoFun5
- Tu/e email: f.wu@student.tue.nl