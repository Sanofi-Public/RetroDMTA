# Utils
import json
import pickle
from pathlib import Path
import datetime

# Data
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# Cheminformatics
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity 
from rdkit.DataStructs.cDataStructs import ExplicitBitVect  
from skfp.fingerprints import ECFPFingerprint
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MolFragmentToSmiles

def save_pickle(obj, file_path):
    """
    Save an object to a pickle file.

    Parameters:
        obj: The Python object to pickle.
        file_path (str): The file path where the pickle file will be saved.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        print(f"❌ Error saving object to {file_path}: {e}")

def load_pickle(file_path):
    """
    Load an object from a pickle file.

    Parameters:
        file_path (str): The file path of the pickle file to load.

    Returns:
        The Python object loaded from the pickle file, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    except Exception as e:
        print(f"❌ Error loading object from {file_path}: {e}")
        return None
    
def load_json(file_path):
    """
    Load a JSON file from the given file path.

    Args:
        file_path (str): The path to the JSON file to load.

    Returns:
        dict: The loaded JSON data as a dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to decode JSON. {e}")
    return None


def save_json(data, file_path):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The path to the JSON file to save.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"✅ JSON data successfully saved to {file_path}")
    except Exception as e:
        print(f"❌ Error: Could not save JSON to {file_path}. {e}")


def to_mol(smi):
    return Chem.MolFromSmiles(smi)


def to_smi(mol):
    return Chem.MolToSmiles(mol)


def to_mols(smis):
    return [to_mol(smi) for smi in smis]


def to_smis(mols):
    return [to_smi(mol) for mol in mols]


def round_to_first_of_next_month(dt):
    """
    Round a datetime.date or datetime.datetime to the first day
    of the *following* month at midnight (if dt is datetime).
    """
    year = dt.year
    month = dt.month

    # If it's December, the next month is January of the next year
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1

    # Return new datetime. 
    # Note: if you only need a date object, use `datetime.date(year, month, 1)`.
    return pd.to_datetime(f'{year}-{month}-01')


def sigmoid_HTB(x, LTV, HTV, q=1, HS=0.05, LS=0.05, cutoff=0.05):
    """
    Higher-the-better sigmoid function.
    
    Parameters:
        x (float): The input value.
        LTV (float): Lower asymptotic value.
        HTV (float): Higher asymptotic value.
        q (float): Quantile for determining the sigmoid center, default is 1.
        HS (float): High saturation level, default is 0.05.
        LS (float): Low saturation level, default is 0.05.
        cutoff (float): Minimum value to return, default is 0.05.

    Returns:
        float: Sigmoid function result for HTB.
    """
    quantile_value = np.quantile([LTV, HTV], q=1-q)

    if LTV == HTV:
        if x > HTV:
            return 1
        else:
            return cutoff
    else:
        value = (x - quantile_value) * 10 / (LTV - HTV)
        clipped_value = np.clip(value, -700, 700)
        exp_value = np.exp(clipped_value)
        sigmoid_value = (1 / (1 + exp_value)) * (1 - HS) + LS
    return np.round(sigmoid_value, 4)


def sigmoid_LTB(x, LTV, HTV, q=1, HS=0.05, LS=0.05, cutoff=0.05):
    """
    Lower-the-better sigmoid function.
    
    Parameters:
        x (float): The input value.
        LTV (float): Lower asymptotic value.
        HTV (float): Higher asymptotic value.
        q (float): Quantile for determining the sigmoid center, default is 1.
        HS (float): High saturation level, default is 0.05.
        LS (float): Low saturation level, default is 0.05.
        cutoff (float): Minimum value to return, default is 0.05.

    Returns:
        float: Sigmoid function result for LTB.
    """
    quantile_value = np.quantile([LTV, HTV], q=q)

    if LTV == HTV:
        if x < LTV:
            return 1
        else:
            return cutoff
    else:
        value = (x - quantile_value) * 10 / (HTV - LTV)
        clipped_value = np.clip(value, -700, 700)
        exp_value = np.exp(clipped_value)
        sigmoid_value = (1 / (1 + exp_value)) * (1 - HS) + LS
    return np.round(sigmoid_value, 4)


def sigmoid_INT(x, LTV, HTV, q=1, HS=0.05, LS=0.05, cutoff=0.05):
    """
    Interval sigmoid function combining HTB and LTB.
    
    Parameters:
        x (float): The input value.
        LTV (float): Lower asymptotic value.
        HTV (float): Higher asymptotic value.
        q (float): Quantile for determining the sigmoid center, default is 1.
        HS (float): High saturation level, default is 0.05.
        LS (float): Low saturation level, default is 0.05.
        cutoff (float): Minimum value to return, default is 0.05.

    Returns:
        float: Sigmoid function result for intermediate values.
    """
    mean_value = (LTV + HTV) / 2
    if x < mean_value:
        return sigmoid_HTB(x, LTV, mean_value, q, HS, LS, cutoff)
    else:
        return sigmoid_LTB(x, mean_value, HTV, q, HS, LS, cutoff)
    

def sum_values(row, value_dict):
    """
    Calculate the sum of values from a dictionary where the corresponding keys exist in the row 
    and are not NaN.

    Parameters:
        row (dict): Row data containing potential keys from the value dictionary.
        value_dict (dict): Dictionary with values to sum.

    Returns:
        int: The total sum of values.
    """
    total = 0
    for key, value in value_dict.items():
        # Check if the key exists in the row and the value is not NaN
        if key in row and not pd.isna(row[key]):
            total += value
    return total


def normalize_and_score_property(df, prop, bp_df, sigmoid_funcs):
    """
    For a given property, extract its parameters, normalize its values 
    using the corresponding sigmoid function, and compute weighted scores.
    
    The function adds three new columns to the DataFrame:
      - normalized_{prop}
      - geometric_{prop} (normalized value raised to the power of its weight)
      - arithmetic_{prop} (normalized value multiplied by its weight)
    
    Parameters:
      df           : DataFrame to operate on.
      prop         : The property (column) name.
      bp_df        : DataFrame containing property parameters (TREND, LTV, HTV, WEIGHT).
      sigmoid_funcs: Dictionary mapping trend values to sigmoid functions.
    
    Returns:
      A tuple of column names for (normalized, geometric, arithmetic) scores.
    """
    params = bp_df.loc[bp_df['PROPERTIES'] == prop, ['TREND', 'LTV', 'HTV', 'WEIGHT']].squeeze()
    trend, ltv, htv, weight = params['TREND'], params['LTV'], params['HTV'], params['WEIGHT']
    sigmoid_fn = sigmoid_funcs[trend]
    
    norm_col = f'normalized_{prop}'
    geom_col = f'geometric_{prop}'
    arith_col = f'arithmetic_{prop}'
    
    # Normalize property values using the sigmoid function
    df[norm_col] = df[prop].apply(lambda x: sigmoid_fn(x, ltv, htv, q=1))
    
    # Calculate weighted scores
    df[geom_col] = df[norm_col] ** weight
    df[arith_col] = df[norm_col] * weight

    return norm_col, geom_col, arith_col


def select_best_smiles(input_df, score_col, fraction, annotated_smiles):
    """
    Select SMILES strings for which the given score exceeds the (1 - fraction) quantile.
    
    Parameters:
      input_df        : DataFrame containing the score column and 'SMILES'.
      score_col       : The name of the score column to use for thresholding.
      fraction        : The fraction for the quantile threshold (e.g. 0.10 for 10%).
      annotated_smiles: List of annotated SMILES to include.
    
    Returns:
      A list of unique SMILES strings.
    """
    threshold = input_df[score_col].quantile(1 - fraction)
    smiles = input_df.loc[input_df[score_col] >= threshold, 'SMILES'].tolist()
    return list(set(smiles + annotated_smiles))


def convert_to_BitVect(array):
    """
    Converts a binary array into an RDKit ExplicitBitVect object.

    Parameters:
        array (list or np.ndarray): Binary array representing a fingerprint.

    Returns:
        ExplicitBitVect: RDKit explicit bit vector.
    """
    bitvect = ExplicitBitVect(len(array))
    for idx, val in enumerate(array):
        if val:  # Set bit if value is True or non-zero
            bitvect.SetBit(idx)
    return bitvect


def compute_similarity_matrix(smiles):
    """
    Computes the Tanimoto similarity matrix for a set of fingerprints.

    Parameters:
        smiles (list): List of SMILES strings corresponding to the fingerprints.
        fingerprints (list): List of binary fingerprints to be converted to ExplicitBitVect.

    Returns:
        pd.DataFrame: DataFrame representing the similarity matrix, indexed by SMILES.
    """
    # Compute binary ECFP
    fingerprints = ECFPFingerprint(fp_size=2048, radius=2, include_chirality=False).transform(smiles)
    # Convert binary fingerprints to ExplicitBitVect
    fingerprints = [convert_to_BitVect(fp) for fp in fingerprints]
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))  # Initialize similarity matrix

    # Compute Tanimoto similarity matrix
    for i in range(n):
        if fingerprints[i] is not None:
            similarities = BulkTanimotoSimilarity(fingerprints[i], fingerprints[i:])
            similarity_matrix[i, i:] = similarities  # Fill upper triangle
            similarity_matrix[i:, i] = similarities  # Mirror to lower triangle

    # Create a DataFrame for the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix, index=smiles, columns=smiles)

    return similarity_df


def get_r_groups(smiles1, smiles2, scaffold_smiles, cache_mols, cache_scaffolds):
    """
    Extract R-groups from two molecules based on a shared scaffold.
    
    Parameters:
        smiles1 (str): SMILES of the first molecule.
        smiles2 (str): SMILES of the second molecule.
        scaffold_smiles (str): SMILES of the scaffold.

    Returns:
        diff_mol1 (Mol): R-group of the first molecule.
        diff_mol2 (Mol): R-group of the second molecule.
    """
    mol1, mol2, scaffold = cache_mols[smiles1], cache_mols[smiles2], cache_scaffolds[scaffold_smiles]

    # Find substructure matches
    match1 = mol1.GetSubstructMatch(scaffold)
    match2 = mol2.GetSubstructMatch(scaffold)

    if not match1 or not match2:
        # print("No match found.")
        return None, None

    # Remove scaffold atoms from mol1
    emol1 = Chem.EditableMol(Chem.Mol(mol1))
    for idx in sorted(list(match1), reverse=True):
        emol1.RemoveAtom(idx)
    diff_mol1 = emol1.GetMol()

    # Remove scaffold atoms from mol2
    emol2 = Chem.EditableMol(Chem.Mol(mol2))
    for idx in sorted(list(match2), reverse=True):
        emol2.RemoveAtom(idx)
    diff_mol2 = emol2.GetMol()

    return diff_mol1, diff_mol2


def load_common_config(filepath):
    """
    Load the common configuration from a JSON file.

    Parameters:
        filepath (str): The path to the JSON configuration file. 
                        Defaults to '../../../data/common/simulations_config.json'.

    Returns:
        dict: The configuration dictionary loaded from the JSON file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    config_path = Path(filepath)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {filepath}")
    
    with config_path.open('r', encoding='utf-8') as file:
        config = json.load(file)
    
    return config


def get_time():
    return (datetime.datetime.now()+datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")

def get_pmas_palette():
    """
    Returns the PMAS color palette as a list of hex color codes.

    Colors:
        - Turquoise: '#25b095'
        - Orange:    '#ed7f19'
        - Blue:      '#2188e9'
        - Red:       '#ec4e3a'
        - Purple:    '#8d35c4'
        - Green:     '#20da70'
        - Pink:      '#e988c0'
        - Yellow:    '#f6cc0c'
        - Beige:     '#c4b99f'
        - Black:     '#2b2726'
    
    Returns:
        List[str]: The list of hex color codes.
    """
    return [
        '#25b095',  # Turquoise
        '#ed7f19',  # Orange
        '#2188e9',  # Blue
        '#ec4e3a',  # Red
        '#8d35c4',  # Purple
        '#20da70',  # Green
        '#e988c0',  # Pink
        '#c4b99f',  # Beige
        '#f6cc0c',  # Yellow
        '#2b2726'   # Black
    ]


def extract_year_month(df: pd.DataFrame):
    """
    Extracts the year and month from the 'DATE' column of a DataFrame and
    adds them as new columns named 'Year' and 'Month'.

    Assumes that the 'DATE' column is of datetime type.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'DATE' column.

    Returns:
        pd.DataFrame: A new DataFrame with added 'Year' and 'Month' columns.
    """
    df = df.copy()  # Avoid modifying the original DataFrame
    df['Year'] = df['DATE'].dt.year
    df['Month'] = df['DATE'].dt.month
    return df


def load_tmap_coord(dataset):
    with open(f'../../data/{dataset}/tmap_coordinates.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    x = loaded_dict['x']
    y = loaded_dict['y']
    s = loaded_dict['s']
    t = loaded_dict['t']

    return x, y, s, t


def convert_number(s):
    if s.startswith("0"):
        return(f"0.{s[1:]}")
    else:
        return s
    

def get_clean_batchsize(raw_BS):
    return convert_number(raw_BS.split('_')[-1])



def get_clean_model(raw_MODEL):

    map_models = {
    'gp' : 'Gaussian Process',
    'rf' : 'Random Forest',
    'xgb' : 'XGBoost',
    'cb' : 'CatBoost',
    'lgbm' : 'LightGBM'
    }

    model_name = raw_MODEL.split('_')[1]
    uncertainty_name = raw_MODEL.split('_')[2]
    return f"{map_models[model_name]} ({uncertainty_name})"


def get_clean_type(raw_TYPE):

    if raw_TYPE == "pure":
        return "Pure"
    else:
        return raw_TYPE.replace('_', '/')
    

def get_clean_strategy(raw_STRATEGY):

    map_strategies = {        
        "random" : "Random",
        "kmeans" : "K-means",
        "kmedoids" : "K-medoids",
        "coverage": "Coverage",
        "spread": "Spread",
        "similarity": "Similarity-to-known",
        "dissimilarity": "Dissimilarity-to-known",
        "best_similarity": "Similarity-to-known-good",
        "best_dissimilarity": "Dissimilarity-to-known-good",
        "desirability": "Desirability",
        "desirability_wAD": "Desirability (with AD)",
        "uncertainty": "Uncertainty",
        "harmonic": "U/D Harmonic",
        "desired_coverage": "Desired coverage",
        "desired_spread": "Desired spread",
        "topsis": "TOPSIS",
        "gra": "GRA",
        "oracle": "Oracle"
        }
    
    if "greediverse" in raw_STRATEGY:
        return f"Greediverse (λ = {convert_number(raw_STRATEGY.split('_')[1])}, t = {convert_number(raw_STRATEGY.split('_')[2])})"
    elif "ratio" in raw_STRATEGY:
        return f"U/D Ratio" 
    else:
        return map_strategies[raw_STRATEGY]
    

def get_kde(x, y, bw_adjust=0.1, gridsize=200, binary=False):

    data = np.vstack([x, y])
    kde = gaussian_kde(data)
    kde.set_bandwidth(bw_method=kde.scotts_factor() * bw_adjust)
    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    xx, yy = np.mgrid[xmin:xmax:gridsize*1j, ymin:ymax:gridsize*1j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid_coords).reshape(xx.shape)
    
    if binary == True:
        density = np.where(density < density.mean(), 0, 1)

    return density


def compute_TMAPOverlap(x, y, project_kde):
    
    kde = get_kde(x, y, binary=True)
    sharp_kde = np.logical_and(kde == 1, project_kde == 1).astype(int)
    
    overlap = sharp_kde.sum() / project_kde.sum()
    
    return overlap


def compute_DistanceCoverage(smis, subset_smis, distance_matrix, threshold=0.65):
    
    small_distance_matrix = distance_matrix.loc[subset_smis, smis]
    
    coverage = (small_distance_matrix <= threshold).any().sum() / distance_matrix.shape[1]
    
    return coverage


def compute_DistanceCoverageAUC(smis, subset_smis, distance_matrix):
    
    small_distance_matrix = distance_matrix.loc[subset_smis, smis]
    
    thresholds = []
    counts = []
    
    for threshold in np.arange(0, 1.05, 0.05):
        count = (small_distance_matrix <= threshold).any().sum()
        thresholds.append(threshold)
        counts.append(count)
        
    counts = np.array(counts)/max(counts)
    
    auc = np.trapz(counts, thresholds)
    
    return auc


def get_bemis_murcko_scaffold(smiles):
    """
    Given a SMILES string, returns the Bemis–Murcko scaffold as a canonical SMILES.
    Returns None if the molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print(f"Warning: Could not parse SMILES '{smiles}'")
        return None
    scaffold = MurckoScaffold.MakeScaffoldGeneric(mol)
    if not scaffold:
        return None
    # Convert the scaffold molecule back to a canonical SMILES string.
    return Chem.MolToSmiles(scaffold, canonical=True)


def compute_BMScaffoldCoverage(smis, subset_smis):
    scaffolds = np.unique([get_bemis_murcko_scaffold(smi) for smi in smis])
    subset_scaffolds = np.unique([get_bemis_murcko_scaffold(smi) for smi in subset_smis])
    coverage_BM = len(subset_scaffolds) / len(scaffolds)
    return coverage_BM


def merge(mol, marked, current_set):
    """
    Recursively add any marked atoms that are connected to the atoms in current_set.
    """
    new_atoms = set()
    for idx in current_set:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                new_atoms.add(jdx)
    if new_atoms:
        # Continue merging with the newly found atoms.
        merge(mol, marked, new_atoms)
        current_set.update(new_atoms)


# --- SMARTS Patterns ---
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('C=,#C')
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')
PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)


def identify_functional_groups_from_mol(mol):
    """
    Identify functional groups in an RDKit molecule.
    
    The algorithm marks all heteroatoms (any atom not C or H) and also marks additional
    atoms based on several SMARTS patterns. It then merges connected marked atoms into groups
    and returns the canonical SMILES for each group.
    
    Args:
        mol (rdkit.Chem.Mol): The molecule to analyze.
    
    Returns:
        list of str: A list of functional group SMILES.
    """
    marked = set()
    
    # Mark all heteroatoms (non-carbon, non-hydrogen)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6, 1):
            marked.add(atom.GetIdx())
    
    # Mark additional atoms matching specific patterns
    for patt in PATT_TUPLE:
        for match in mol.GetSubstructMatches(patt):
            marked.update(match)
    
    # Merge connected marked atoms into groups
    groups = []
    while marked:
        group = {marked.pop()}
        merge(mol, marked, group)
        groups.append(group)
    
    # Convert each group of atoms into a canonical SMILES fragment
    fgs = []
    for group in groups:
        fg_smiles = MolFragmentToSmiles(mol, list(group), canonical=True)
        fgs.append(fg_smiles)
    
    return fgs


def compute_functional_groups(smiles_list, unique=True):
    """
    Compute functional groups for a list of SMILES strings.
    
    For each SMILES, the molecule is constructed and analyzed to identify functional groups.
    The function returns a flattened list of functional group SMILES. Optionally, duplicates 
    can be removed.
    
    Args:
        smiles_list (list of str): List of input SMILES strings.
        unique (bool): If True, return only unique functional groups (default: True).
    
    Returns:
        list of str: A list of functional group SMILES.
    """
    all_fgs = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            # Skip invalid SMILES
            continue
        fgs = identify_functional_groups_from_mol(mol)
        all_fgs.extend(fgs)
    
    if unique:
        # Remove duplicates while preserving order
        seen = set()
        unique_fgs = []
        for fg in all_fgs:
            if fg not in seen:
                seen.add(fg)
                unique_fgs.append(fg)
        return unique_fgs
    else:
        return all_fgs

def compute_FunctionalGroupsCoverage(smis, subset_smis):
    fgs = compute_functional_groups(smis)
    subset_fgs = compute_functional_groups(subset_smis)
    coverage_FG = len(subset_fgs) / len(fgs)
    return coverage_FG


def compute_Diversity(distance_matrix):
    """
    Given an n x n distance matrix D (with D[i, i] = 0),
    compute the Diversity metric:

      Diversity(S; d) = 1/(n(n-1)) * ∑_{x,y in S, x≠y} d(x,y)
    """
    
    if isinstance(distance_matrix, pd.DataFrame):
        D = distance_matrix.values
    else:
        D = distance_matrix
    
    n = D.shape[0]

    # Sum of upper-triangular (i < j) distances:
    sum_upper = np.sum(np.triu(D, k=1))
    # Sum over *all* x ≠ y (ordered pairs) is twice that:
    sum_all = 2.0 * sum_upper

    # Diversity
    Diversity = (1.0 / (n * (n - 1))) * sum_all

    return Diversity


def compute_SumDiversity(distance_matrix):
    """
    Given an n x n distance matrix D (with D[i, i] = 0),
    compute the SumDiversity metric:

      SumDiversity(S; d)   = 1/(n - 1) * ∑_{x,y in S, x≠y} d(x,y)
    """
    
    if isinstance(distance_matrix, pd.DataFrame):
        D = distance_matrix.values
    else:
        D = distance_matrix
        
    n = D.shape[0]

    # Sum of upper-triangular (i < j) distances:
    sum_upper = np.sum(np.triu(D, k=1))
    # Sum over *all* x ≠ y (ordered pairs) is twice that:
    sum_all = 2.0 * sum_upper

    # SumDiversity
    SumDiversity = (1.0 / (n - 1)) * sum_all

    return SumDiversity


def compute_Diameter(distance_matrix):
    """
    Given an n x n distance matrix D (with D[i, i] = 0),
    compute the Diameter metric:

      Diameter(S; d)       = max_{x,y in S, x≠y} d(x,y)
    """
    
    if isinstance(distance_matrix, pd.DataFrame):
        D = distance_matrix.values
    else:
        D = distance_matrix
    
    # Diameter(S; d) = max over off-diagonal entries
    temp = D.copy()
    np.fill_diagonal(temp, -np.inf)  # ignore self-distances
    Diameter = np.max(temp)
    
    return Diameter


def compute_SumDiameter(distance_matrix):
    """
    Given an n x n distance matrix D (with D[i, i] = 0),
    compute the SumDiameter metric:

      SumDiameter(S; d)    = ∑_{x in S} [max_{y in S, y≠x} d(x,y)]
    """
    
    if isinstance(distance_matrix, pd.DataFrame):
        D = distance_matrix.values
    else:
        D = distance_matrix

    # SumDiameter(S; d) = sum_x [ max_{y≠x} d(x,y) ]
    temp = D.copy()
    np.fill_diagonal(temp, -np.inf)
    rowwise_max = np.max(temp, axis=1)
    SumDiameter = np.sum(rowwise_max)
    
    return SumDiameter


def compute_Bottleneck(distance_matrix):
    """
    Given an n x n distance matrix D (with D[i, i] = 0),
    compute the Bottleneck metric:

      Bottleneck(S; d)     = min_{x,y in S, x≠y} d(x,y)
    """
    
    if isinstance(distance_matrix, pd.DataFrame):
        D = distance_matrix.values
    else:
        D = distance_matrix
    
    # Bottleneck(S; d) = min over off-diagonal entries
    temp = D.copy()
    np.fill_diagonal(temp, np.inf)
    Bottleneck = np.min(temp)
    
    return Bottleneck

def compute_SumBottleneck(distance_matrix):
    """
    Given an n x n distance matrix D (with D[i, i] = 0),
    compute the SumBottleneck metric:

      SumBottleneck(S; d)  = ∑_{x in S} [min_{y in S, y≠x} d(x,y)]
    """
    
    if isinstance(distance_matrix, pd.DataFrame):
        D = distance_matrix.values
    else:
        D = distance_matrix

    # SumBottleneck(S; d) = sum_x [ min_{y≠x} d(x,y) ]
    temp = D.copy()
    np.fill_diagonal(temp, np.inf)
    rowwise_min = np.min(temp, axis=1)
    SumBottleneck = np.sum(rowwise_min)
    
    return SumBottleneck


def sphere_exclusion(distance_matrix, threshold):
    # Convert DataFrame to numpy array if needed
    if isinstance(distance_matrix, pd.DataFrame):
        distance_matrix = distance_matrix.values

    # Create a boolean connectivity matrix (neighbors within threshold)
    connectivity = distance_matrix < threshold
    # Remove self-connections
    np.fill_diagonal(connectivity, False)

    n = connectivity.shape[0]
    selected_nodes = []
    forbidden = np.zeros(n, dtype=bool)
    
    # Greedily select nodes: if a node is not forbidden, select it
    # and mark all its neighbors as forbidden.
    for i in range(n):
        if not forbidden[i]:
            selected_nodes.append(i)
            # Mark neighbors as forbidden.
            forbidden |= connectivity[i]
    
    return selected_nodes

def compute_Circles(distance_matrix, threshold):
    return len(sphere_exclusion(distance_matrix, threshold))


def get_ring_systems_from_mol(mol, include_spiro=False):
    """
    Extract ring systems from an RDKit molecule.
    
    A ring system is defined as one or more rings that share atoms.
    If include_spiro is False, rings that share only a single atom (spiro connections)
    will not be merged.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule to analyze.
        include_spiro (bool): Whether to merge rings sharing a single atom.

    Returns:
        list of str: A list of canonical SMILES strings, one for each ring system.
    """
    ri = mol.GetRingInfo()
    systems = []  # each system is a set of atom indices

    for ring in ri.AtomRings():
        ring_atoms = set(ring)
        new_systems = []
        # Merge with any existing system that shares atoms
        for system in systems:
            common_atoms = ring_atoms.intersection(system)
            # Merge if there's any common atom (and, if not including spiro, require >1 in common)
            if common_atoms and (include_spiro or len(common_atoms) > 1):
                ring_atoms = ring_atoms.union(system)
            else:
                new_systems.append(system)
        new_systems.append(ring_atoms)
        systems = new_systems

    ring_smiles = []
    for system in systems:
        # Convert the set of atom indices to a canonical SMILES fragment.
        frag_smiles = MolFragmentToSmiles(mol, list(system), canonical=True)
        ring_smiles.append(frag_smiles)
    return ring_smiles


def compute_ring_systems(smiles_list, include_spiro=False, unique=True):
    """
    Compute ring systems for a list of SMILES strings.
    
    For each SMILES string, the molecule is constructed and its ring systems are extracted.
    The function returns a flattened list of ring system SMILES. Optionally, duplicates can be removed.

    Args:
        smiles_list (list of str): List of input SMILES strings.
        include_spiro (bool): Whether to merge spiro rings (rings sharing a single atom).
        unique (bool): If True, return only unique ring system SMILES (default: True).

    Returns:
        list of str: A list of ring system SMILES.
    """
    all_ring_systems = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Skip invalid SMILES
            continue
        try:
            rings = get_ring_systems_from_mol(mol, include_spiro=include_spiro)
            all_ring_systems.extend(rings)
        except Exception:
            # Skip molecules where ring system extraction fails
            continue

    if unique:
        seen = set()
        unique_rings = []
        for ring in all_ring_systems:
            if ring not in seen:
                seen.add(ring)
                unique_rings.append(ring)
        return unique_rings
    else:
        return all_ring_systems

def compute_RingSystemsCoverage(smis, subset_smis):
    ring_systems = compute_ring_systems(smis, include_spiro=False, unique=True)
    subset_ring_systems = compute_ring_systems(subset_smis, include_spiro=False, unique=True)
    coverage_RS = len(subset_ring_systems) / len(ring_systems)
    return coverage_RS


def is_cliff(x, std):
    if pd.isna(x):
        return np.nan
    elif x > 2 * std:
        return 1
    else:
        return 0