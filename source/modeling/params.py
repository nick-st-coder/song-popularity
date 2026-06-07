def get_lgbm_params(trial):
    return {
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample':trial.suggest_float('subsample', 0.5, 1.0),
        'learning_rate':trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth':trial.suggest_int('max_depth', 5, 12),
        'n_estimators':trial.suggest_int('n_estimators', 100, 1200),
        'num_leaves':trial.suggest_int('num_leaves', 15, 255, log=True),
        'reg_alpha':trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples':trial.suggest_int('min_child_samples', 5, 100),

        'objective':'regression',
        'random_state':42,
        'verbosity':-1
    }

def get_rf_params(trial):
    return {
        'n_estimators':trial.suggest_int('n_estimators', 200, 800),
        'max_depth':trial.suggest_int('max_depth', 4, 8),
        'min_samples_leaf':trial.suggest_int("min_samples_leaf", 20, 80),
        'min_samples_split':trial.suggest_int("min_samples_split", 30, 120),
        'max_features':trial.suggest_categorical(
            "max_features", ['sqrt', 'log2', None]
        ),

        'random_state':4,
        'verbose':1,
        'n_jobs':-1
    }