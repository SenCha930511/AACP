import modules
from config import PruningConfig, LoRaConfig, EvaluateConfig, WandappConfig, ActivationAnalysisConfig


class ActivationAwarePruning():
    """
    實現 Activation-Aware Calibration Data Selection for Pruning 演算法的核心類別。
    提供模型評估、LoRA 微調、WANDA 剪枝等功能。
    """
    def __init__(self, model_path):
        self.set_model_path(model_path)

    def set_model_path(self, model_path):
        self.model_path = model_path

    def evaluate(self, config: EvaluateConfig):
        modules.evaluate(config, self.model_path)

    def lora_finetune(self, config: LoRaConfig):
        modules.lora_finetune(config, self.model_path)

    def prune_wanda(self, config: PruningConfig):
        return modules.prune_wanda(config, self.model_path)

    def evaluate_model_sparsity(self):
        return modules.evaluate_model_sparsity(self.model_path)

    def prune_wandapp(self, config: WandappConfig):
        return modules.prune_wandapp(config, self.model_path)

    def analyze_activation(self, config: ActivationAnalysisConfig):
        return modules.compare_activation_coverage(config)

    def compare_global_activation_coverage(self, config: ActivationAnalysisConfig):
        return modules.compare_global_activation_coverage(config)

    def compare_activation_gradient(self, config: ActivationAnalysisConfig):
        return modules.compare_activation_gradient(config)
