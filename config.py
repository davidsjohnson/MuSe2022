import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# adjust your paths here. Recommended to keep it that way in order not to run into git conflicts
DATA_PATH = 'data/'
BASE_PATH = '../MuSe2022'

PATH_TO_FEATURES = {
    'humor': os.path.join(DATA_PATH, 'c1_muse_humor_2022/feature_segments'),
    'reaction': os.path.join(DATA_PATH, 'c2_muse_reaction_2022/feats'),
    'stress': os.path.join(DATA_PATH, 'c3_muse_stress_2022/feature_segments'),
    'tl_stress': os.path.join(DATA_PATH, 'tl_stress_2022/feature_segments')
}

# humor is labelled every 2s, but features are extracted every 500ms
N_TO_1_TASKS = {'humor', 'reaction'}

ACTIVATION_FUNCTIONS = {
    'humor': torch.nn.Sigmoid,
    'reaction': torch.nn.Sigmoid,
    'stress':torch.nn.Tanh,
    'tl_stress': torch.nn.Tanh, # TODO compare Tanh (with label norm) with no activation (no label norm)
}

NUM_TARGETS = {
    'humor': 1,
    'reaction': 7,
    'stress': 1,
    'tl_stress': 1
}


PATH_TO_LABELS = {
    'humor': os.path.join(DATA_PATH, 'c1_muse_humor_2022/label_segments'),
    'reaction': os.path.join(DATA_PATH, 'c2_muse_reaction_2022'),
    'stress': os.path.join(DATA_PATH, 'c3_muse_stress_2022/label_segments'),
    'tl_stress': os.path.join(DATA_PATH, 'tl_stress_2022/label_segments')
}

PATH_TO_METADATA = {
    'humor': os.path.join(DATA_PATH, 'c1_muse_humor_2022/metadata'),
    'reaction':os.path.join(DATA_PATH, 'c2_muse_reaction_2022'),
    'stress': os.path.join(DATA_PATH, 'c3_muse_stress_2022/metadata'),
    'tl_stress': os.path.join(DATA_PATH, 'tl_stress_2022/metadata'),
}

PARTITION_FILES = {task: os.path.join(path_to_meta, 'partition.csv') for task,path_to_meta in PATH_TO_METADATA.items()}

REACTION_LABELS = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise']

OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
LOG_FOLDER = os.path.join(OUTPUT_PATH, 'log_muse')
DATA_FOLDER = os.path.join(OUTPUT_PATH, 'data_muse')
MODEL_FOLDER = os.path.join(OUTPUT_PATH, 'model_muse')
PREDICTION_FOLDER = os.path.join(OUTPUT_PATH, 'prediction_muse')
