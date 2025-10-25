#!/bin/bash
#SBATCH --job-name=aacp
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=aacp-%j.out
#SBATCH --error=aacp-%j.err
#SBATCH --account=
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=acs111132@gm.ntcu.edu.tw

echo "作業於 $(hostname) 在 $(date) 開始執行"
echo "目前的工作目錄 (變更前): $(pwd)"


cd $HOME/AACP/
echo "工作目錄已變更為: $(pwd)"

source aacp_env/bin/activate 

echo "虛擬環境激活後的 Python 直譯器路徑:"
which python
echo "虛擬環境激活後的 Python 版本:"
python --version

echo "正在運行 test.py..."
python test.py

echo "作業於 $(date) 完成。"
