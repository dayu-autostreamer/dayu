import torch
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.join(current_dir, '..', '..', 'once-for-all')

def get_ofa_supernet_mbv3_w10():
    return torch.hub.load(repo_dir, 'ofa_supernet_mbv3_w10', pretrained=False, force_reload=False, source='local')

def get_ofa_supernet_mbv3_w12():
    return torch.hub.load(repo_dir, 'ofa_supernet_mbv3_w12', pretrained=False, force_reload=False, source='local')

def get_ofa_supernet_resnet50():
    return torch.hub.load(repo_dir, 'ofa_supernet_resnet50', pretrained=False, force_reload=False, source='local')


def get_architecture_dict(ofa_supernet_name: str):
    assert ofa_supernet_name in ["ofa_supernet_mbv3_w10", "ofa_supernet_mbv3_w12", "ofa_supernet_resnet50"]
    if ofa_supernet_name == "ofa_supernet_mbv3_w10":
        return {
            "ks": {
                "length": 20,
                "choices": [3, 5, 7],
            },
            "e": {
                "length": 20,
                "choices": [3, 4, 6],
            },
            "d": {
                "length": 5,
                "choices": [2, 3, 4],
            }
        }
    elif ofa_supernet_name == "ofa_supernet_mbv3_w12":
        return {
            "ks": {
                "length": 20,
                "choices": [3, 5, 7],
            },
            "e": {
                "length": 20,
                "choices": [3, 4, 6],
            },
            "d": {
                "length": 5,
                "choices": [2, 3, 4],
            }
        }
    elif ofa_supernet_name == "ofa_supernet_resnet50":
        return {
            "d": {
                "length": 5,
                "choices": [0, 1, 2],
            },
            "e": {
                "length": 18,
                "choices": [0.2, 0.25, 0.35],
            },
            "w": {
                "length": 6,
                "choices": [0, 1, 2],
            }
        }

def get_max_net_config(ofa_supernet_name: str):
    assert ofa_supernet_name in ["ofa_supernet_mbv3_w10", "ofa_supernet_mbv3_w12", "ofa_supernet_resnet50"]
    if ofa_supernet_name == "ofa_supernet_mbv3_w10":
        return {
            "ks": [7] * 20,
            "e": [6] * 20,
            "d": [4] * 5
        }
    elif ofa_supernet_name == "ofa_supernet_mbv3_w12":
        return {
            "ks": [7] * 20,
            "e": [6] * 20,
            "d": [4] * 5
        }
    elif ofa_supernet_name == "ofa_supernet_resnet50":
        return {
            "d": [2] * 5,
            "e": [0.35] * 18,
            "w": [2] * 6
        }
    
def get_min_net_config(ofa_supernet_name: str):
    assert ofa_supernet_name in ["ofa_supernet_mbv3_w10", "ofa_supernet_mbv3_w12", "ofa_supernet_resnet50"]
    if ofa_supernet_name == "ofa_supernet_mbv3_w10":
        return {
            "ks": [3] * 20,
            "e": [3] * 20,
            "d": [2] * 5
        }
    elif ofa_supernet_name == "ofa_supernet_mbv3_w12":
        return {
            "ks": [3] * 20,
            "e": [3] * 20,
            "d": [2] * 5
        }
    elif ofa_supernet_name == "ofa_supernet_resnet50":
        return {
            "d": [0] * 5,
            "e": [0.2] * 18,
            "w": [0] * 6
        }