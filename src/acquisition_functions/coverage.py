from multiprocessing import Pool
from functools import partial
import numpy as np

def coverage_objective_function(smiles_selected, smiles_train, smiles_pool, fingerprint_df):
    
    selected_fps = np.vstack(fingerprint_df[fingerprint_df['SMILES'].isin(smiles_selected)]['fingerprint'].values)
    pool_fps = np.vstack(fingerprint_df[fingerprint_df['SMILES'].isin(smiles_pool)]['fingerprint'].values)
    train_fps = np.vstack(fingerprint_df[fingerprint_df['SMILES'].isin(smiles_train)]['fingerprint'].values)

    F_sampled = train_fps.sum(
        axis=0) + selected_fps.sum(axis=0)
    F_full = train_fps.sum(axis=0) + pool_fps.sum(axis=0)

    N_sampled = len(smiles_selected) + len(train_fps)
    N_full = len(pool_fps) + len(train_fps)
    P_sampled = N_sampled / N_full

    P_smooth = (F_sampled + 1) / (F_full + 1/P_sampled)
    C_base = - np.log(P_smooth/P_sampled)
    p1 = F_sampled / N_sampled
    p2 = 1 - P_sampled

    old_settings = np.seterr(divide='ignore', invalid='ignore')
    H_sampled = -(p1 * np.log(p1) + p2 * np.log(p2)) / np.log(2)
    H_sampled = np.where(F_sampled == N_sampled, 0, H_sampled)
    H_sampled = np.nan_to_num(x=H_sampled, nan=0)
    np.seterr(**old_settings)

    C_final = []

    
    for i, Ci in enumerate(C_base):

        if (Ci > 0) and (p1[i] > 1/2):
            C_final.append(Ci*(2-H_sampled[i]))
        else:
            C_final.append(Ci*H_sampled[i])

    C_final = np.array(C_final)

    score = np.multiply(selected_fps, C_final).sum(axis=1).sum()

    return score

def compute_candidate(smile, smiles_selected, smiles_train, smiles_pool, mhfp_df):
    temp_selected = smiles_selected + [smile]
    coverage_score = coverage_objective_function(temp_selected, smiles_train, smiles_pool, mhfp_df)
    return (smile, coverage_score)

def coverage(self, batch_size):

    if len(self.pool_df) > batch_size:

        smiles_selected = []
        remaining_smiles = set(self.smiles_pool)
        smiles_train = self.smiles_train
        smiles_pool = self.smiles_pool
        mhfp_df = self.mhfp_df

        while len(smiles_selected) < batch_size and remaining_smiles:
            # Create a partial function with the current smiles_selected
            partial_compute_candidate = partial(
                compute_candidate,
                smiles_selected=smiles_selected.copy(),
                smiles_train=smiles_train,
                smiles_pool=smiles_pool,
                mhfp_df=mhfp_df
            )

            with Pool() as pool:
                # Map the function over the remaining_smiles in parallel
                candidates = pool.map(partial_compute_candidate, remaining_smiles)

            # Find the best candidate based on the computed scores
            best_candidate = max(candidates, key=lambda x: x[1])
            smiles_selected.append(best_candidate[0])
            remaining_smiles.remove(best_candidate[0])
    else:
        smiles_selected = self.pool_df['SMILES'].values
    return smiles_selected