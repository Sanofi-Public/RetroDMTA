import numpy as np
import pandas as pd
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint


def compute_fingerprints(smiles, fingerprints=[
                     ECFPFingerprint(fp_size=2048, radius=2, include_chirality=True, count=True, n_jobs=1),
                     MACCSFingerprint(count=True, n_jobs=1),
                     ]):
    fps = []
    for fingerprint in fingerprints:
        fp = fingerprint.transform(smiles)
        fps.append(fp)
    return np.hstack(fps)


def get_fingerprint_dataframe(smiles, fingerprints):
    fps = compute_fingerprints(smiles, fingerprints)  
    return pd.DataFrame({'SMILES': smiles, 'fingerprint': [fp for fp in fps]})