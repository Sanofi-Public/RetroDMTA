
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


def convert_number(s):
    if s.startswith("0"):
        return(f"0.{s[1:]}")
    else:
        return s
    

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
        return f"Ratio ({convert_number(raw_STRATEGY.split('_')[1])})" 
    else:
        return map_strategies[raw_STRATEGY]


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


def get_clean_batchsize(raw_BS):
    return convert_number(raw_BS.split('_')[-1])


def get_clean_exploration_metric(metric):

    map_metrics = {
        "exploration_bm_scaffold_coverage" : "Coverage Bemis-Murcko Scaffold",
        "exploration_fg_coverage" : "Coverage Functional Groups",
        "exploration_rs_coverage" : "Coverage Ring Systems",
        "exploration_neighborhood_coverage" : "Neighborhood Coverage ",
        "exploration_neighborhood_coverage_auc" : "Neighborhood Coverage AUC",
        "exploration_tmap_coverage" : "TMAP Coverage",
        "exploration_circles" : "#Circles",
        "exploration_diversity" : "Internal Diversity",
        "exploration_diameter" : "Diameter",
        "exploration_bottleneck" : "Bottleneck",
        "exploration_sumdiversity" : "SumDiversity",
        "exploration_sumdiameter" : "SumDiameter",
        "exploration_sumbottleneck" : "SumBottleneck",
    }

    return map_metrics[metric]





