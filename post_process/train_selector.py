import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.tree import export_text

# ==========================================
# CONFIGURATION
# ==========================================
LOGS_FILE = 'results_analysis/logs_metrics.csv'
FEATURES_FILE = '../all_features.csv'

# STRATEGY 1: Remove Solver 6 (Low support/Noise)
TARGET_SOLVERS = [5, 7, 8]

# Trade-off Weights
WEIGHT_QUALITY = 0.5
WEIGHT_TIME = 0.5

# STRATEGY 2: Significance Threshold
# If the "best" solver beats the default by less than 5%, we stick with the Default.
SIGNIFICANCE_THRESHOLD = 0.05
DEFAULT_SOLVER = 5

# SELECTED FEATURES (~30 High-Value Features)
SELECTED_FEATURES = [
    # --- HEURISTICS & PROBES ---
    'heur_h_ff_init', 'heur_h_add_init', 'heur_h_max_init',
    'heur_relaxed_plan_length', 'heur_helpful_actions_proxy', 'heur_goal_count',
    
    # --- LANDMARKS ---
    'lm_lm_num_landmarks', 'lm_lm_num_orderings_total', 'lm_lm_num_reasonable_orders',
    'lm_landmarks_per_goal',
    
    # --- CAUSAL GRAPH (Structure) ---
    'sas_cg_num_edges', 'sas_cg_edge_density',
    'sas_cg_max_out_degree', 'sas_cg_num_sinks', 'sas_cg_num_isolated',
    
    # --- SAS+ VARIABLES (Size) ---
    'fd_sas_num_vars', 'fd_sas_num_operators', 'fd_sas_num_mutex_groups',
    'fd_sas_num_axioms', 'fd_sas_var_domain_mean', 'fd_sas_total_facts_proxy',
    
    # --- PDDL LOGIC ---
    'pddl_num_objects', 'pddl_num_predicates', 'pddl_has_conditional_effects',
    'pddl_has_derived_predicates', 'pddl_has_negative_preconditions',
    
    # --- OPERATOR COMPLEXITY ---
    'fd_sas_op_effects_mean', 'fd_sas_op_prevail_mean', 'sas_dtg_trans_mean',
    'fd_translate_seconds'
]

def extract_rules(model, feature_names):
    """
    Extracts human-readable rules from the first tree of the Random Forest.
    """
    print("\n" + "="*40)
    print("INSIGHTS & CONCLUSIONS (Rule Extraction)")
    print("="*40)
    
    print(f"Total features used: {len(feature_names)}")

    # Check if the model is a GridSearchCV object or the Classifier itself
    if hasattr(model, 'best_estimator_'):
        forest = model.best_estimator_
    else:
        forest = model

    print("--- Logic Sample (from Tree #0) ---")
    print("Note: This is one tree out of many, but it represents the general logic.\n")
    
    # Extract rules from the first estimator (Tree)
    one_tree = forest.estimators_[0]
    
    # Limit depth to 3 levels for readability
    tree_rules = export_text(one_tree, feature_names=feature_names, max_depth=3)
    print(tree_rules)
    
    print("\n--- How to read this? ---")
    print("1. 'feature <= value': The condition to check.")
    print("2. 'class: X': The prediction if that condition is met.")
    print("3. Use this to deduce things like: 'High edge density -> Solver 8'")

def train_smart_selector():
    print("1. Loading datasets...")
    try:
        base_path = os.path.dirname(__file__)
        df_logs = pd.read_csv(os.path.join(base_path, LOGS_FILE))
        df_features = pd.read_csv(os.path.join(base_path, FEATURES_FILE))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # -------------------------------------------------------
    # STEP 2: CALCULATE SCORES & ROBUST LABELS
    # -------------------------------------------------------
    print("2. Calculating best solver per problem...")
    
    # Filter logs
    df_logs = df_logs[df_logs['solver'].isin(TARGET_SOLVERS)]
    df_solved = df_logs[df_logs['solved'] == 1].copy()
    df_solved['total_time'] = df_solved['total_time'].replace(0, 0.001)
    
    # Virtual Best Solver Stats
    best_metrics = df_solved.groupby('problem').agg({
        'plan_cost': 'min',
        'total_time': 'min'
    }).rename(columns={'plan_cost': 'min_cost', 'total_time': 'min_time'})

    df_scored = df_solved.merge(best_metrics, on='problem')
    
    # Normalize Scores
    df_scored['quality_score'] = df_scored['min_cost'] / df_scored['plan_cost']
    df_scored['time_score'] = df_scored['min_time'] / df_scored['total_time']
    df_scored['final_score'] = (WEIGHT_QUALITY * df_scored['quality_score']) + \
                               (WEIGHT_TIME * df_scored['time_score'])

    # Determine Winners
    df_scored.sort_values(by=['problem', 'final_score'], ascending=[True, False], inplace=True)
    winners_raw = df_scored.drop_duplicates(subset=['problem'], keep='first')
    
    final_winners = []
    
    # Apply Significance Threshold
    for _, row in winners_raw.iterrows():
        problem = row['problem']
        best_solver = row['solver']
        best_score = row['final_score']
        
        # Check Default Solver Score
        default_entry = df_scored[(df_scored['problem'] == problem) & (df_scored['solver'] == DEFAULT_SOLVER)]
        
        if not default_entry.empty:
            default_score = default_entry.iloc[0]['final_score']
            if (best_score - default_score) < SIGNIFICANCE_THRESHOLD:
                best_solver = DEFAULT_SOLVER
        
        final_winners.append({'problem': problem, 'best_solver': best_solver})

    winners = pd.DataFrame(final_winners)
    print(f"   -> Dataset prepared. Class distribution:\n{winners['best_solver'].value_counts()}")

    # -------------------------------------------------------
    # STEP 3: MERGE & PREPARE DATA
    # -------------------------------------------------------
    print("3. Merging with problem features...")
    dataset = pd.merge(winners, df_features, left_on='problem', right_on='name')
    available_features = [c for c in SELECTED_FEATURES if c in dataset.columns]
    
    X = dataset[available_features]
    y = dataset['best_solver']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # -------------------------------------------------------
    # STEP 4: HYPERPARAMETER TUNING (GRID SEARCH)
    # -------------------------------------------------------
    print("4. Tuning Hyperparameters with Grid Search...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=StratifiedKFold(3),
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"   -> Best Parameters: {grid_search.best_params_}")
    clf_final = grid_search.best_estimator_

    # -------------------------------------------------------
    # STEP 5: EVALUATION
    # -------------------------------------------------------
    print("\n" + "="*40)
    print("MODEL PERFORMANCE REPORT")
    print("="*40)
    
    acc = clf_final.score(X_test, y_test)
    print(f"Global Accuracy: {acc:.2f}")
    
    if len(y_test) > 0:
        y_pred = clf_final.predict(X_test)
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
    
    # Feature Importance
    if hasattr(clf_final, 'feature_importances_'):
        print("\nTop 5 Important Features:")
        importances = clf_final.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(5, len(available_features))):
            print(f"{i+1}. {available_features[indices[i]]} ({importances[indices[i]]:.3f})")

    # -------------------------------------------------------
    # STEP 6: EXTRACT INSIGHTS
    # -------------------------------------------------------
    extract_rules(clf_final, available_features)

if __name__ == "__main__":
    train_smart_selector()