#!/bin/bash
#SBATCH --job-name=wanda_job
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=ent114103

echo "作業於 $(hostname) 在 $(date) 開始執行"
echo "目前的工作目錄 (變更前): $(pwd)"

# 1. 載入系統提供的 miniconda3 模組
# 這會將 miniconda3 的基礎環境添加到 PATH
ml load miniconda3/24.11.1

# 2. **關鍵修改：直接激活您的 Conda 環境**
#    在Slurm腳本中，通常不建議使用 'source .../conda.sh' 和 'conda run'
#    最可靠的方法是直接使用 'conda activate'
#    確保您在 nano5 或 hgpn 節點上已創建 'myenv' 環境
source activate myenv # 直接激活您的 conda 環境

# 3. 進入您的專案目錄 (如果它不在您的提交目錄下)
#    通常，您的腳本會從提交作業的目錄運行，所以如果您的 test.py 就在 AACP 下，
#    這行可能不需要，但為了確保正確，保持也無妨。
cd $HOME/AACP/
echo "工作目錄已變更為: $(pwd)"

# 4. 驗證 Python 版本以確認虛擬環境激活成功
echo "虛擬環境激活後的 Python 直譯器路徑:"
which python
echo "虛擬環境激活後的 Python 版本:"
python --version

# 5. 運行您的 Python 腳本
#    現在 'python' 命令應該指向 'myenv' 環境中的 Python 直譯器了
echo "正在運行 test.py..."
python test.py

echo "作業於 $(date) 完成。"
