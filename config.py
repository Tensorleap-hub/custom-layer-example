import os
import yaml

def get_full_path(path):
    if os.path.isabs(path):
        return path
    elif 'IS_CLOUD' in os.environ and os.path.exists("/nfs"): # SaaS env
        return os.path.join("/nfs", path)
    elif 'GENERIC_HOST_PATH' in os.environ: # OnPrem
        return os.path.join(os.environ['GENERIC_HOST_PATH'], path)
    else:
        return os.path.join(os.path.expanduser('~'), path)
    
def load_config():
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, "config.yaml")
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)


    config["model_path"] = get_full_path(config["model_path"])
    return config

CONFIG = load_config()
    