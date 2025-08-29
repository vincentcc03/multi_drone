import yaml
import os

def load_config(filename):
    """
    根据当前文件路径自动找到配置文件并加载
    filename: 相对于 src/config 的 YAML 文件名，例如 "env_config.yaml"
    """
    # 当前文件所在目录（read_yaml.py）
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # 项目根目录：../../ （因为 read_yaml.py 在 src/utils/）
    project_root = os.path.normpath(os.path.join(current_file_dir, "../../"))

    # 配置文件绝对路径
    config_path = os.path.join(project_root, "src/config", filename)

    # 读取 YAML
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
