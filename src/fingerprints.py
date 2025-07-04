# Data
import numpy as np
import pandas as pd

def compute_fingerprints(smiles, fingerprints):
    fps = []
    for fingerprint in fingerprints:
        fp = fingerprint.transform(smiles)
        fps.append(fp)
    return np.hstack(fps)

def get_fingerprint_dataframe(smiles, fingerprints):
    fps = compute_fingerprints(smiles, fingerprints)  
    return pd.DataFrame({'SMILES': smiles, 'fingerprint': [fp for fp in fps]})