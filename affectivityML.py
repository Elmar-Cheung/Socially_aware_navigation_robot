# affectivityML.py
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

class affectivityML:

    def analyse(self, buffer):
        # ── flatten observations ─────────────────────
        agents      = np.array([o["agents"].flatten() 
                                for o in buffer["observations"]])
        robot_state = np.array([o["robot_state"] 
                                for o in buffer["observations"]])
        X           = np.hstack([agents, robot_state])   # (N, 55)

        valence  = np.array(buffer["valence"])
        intensity  = np.array(buffer["intensity"])
        distance = np.array(buffer["distances"])

        idx = self._stratified_sample(buffer)
        X_balance = X[idx]
        valence_balance = valence[idx]
        intensity_balance = intensity[idx]
        distance_balance = distance[idx]

        self._sanity_check(buffer)
        self._correlation(X_balance, valence_balance, intensity_balance, distance_balance)
        self._mutual_info(X_balance, valence_balance, intensity_balance)
        self._feature_importance(X_balance, valence_balance, intensity_balance)

    def _sanity_check(self, buffer):
        valence  = np.array(buffer["valence"])
        distance = np.array(buffer["distances"])
        
        total    = len(valence)
        neutral  = (valence == 0.0).sum()
        violated = (valence < 0.0).sum()
        
        print(f"Total steps    : {total}")
        print(f"Neutral steps  : {neutral} ({100*neutral/total:.1f}%)")
        print(f"Violated steps : {violated} ({100*violated/total:.1f}%)")
        
    def _stratified_sample(self, buffer, ratio=0.5):
        """
        Balance neutral vs violated steps
        ratio = proportion of violated steps in final sample
        """
        valence  = np.array(buffer["valence"])
        
        violated_idx = np.where(valence < 0.0)[0]
        neutral_idx  = np.where(valence == 0.0)[0]
        
        print(f"Violated: {len(violated_idx)}  Neutral: {len(neutral_idx)}")
        
        # match neutral count to violated count
        n_violated = len(violated_idx)
        neutral_sampled = np.random.choice(
            neutral_idx, 
            size=n_violated,   # equal numbers
            replace=False
        )
        
        combined = np.concatenate([violated_idx, neutral_sampled])
        np.random.shuffle(combined)
        
        return combined

    def _correlation(self, X, valence, intensity, distance):
        """Pearson + Spearman between distance and affect"""
        print("\n── Correlation Analysis ──")
        
        p_val, _ = pearsonr(distance, valence)
        s_val, _ = spearmanr(distance, valence)
        print(f"Distance → Valence  | Pearson: {p_val:.3f}  Spearman: {s_val:.3f}")

        p_aro, _ = pearsonr(distance, intensity)
        s_aro, _ = spearmanr(distance, intensity)
        print(f"Distance → intensity  | Pearson: {p_aro:.3f}  Spearman: {s_aro:.3f}")

    def _mutual_info(self, X, valence, intensity):
        """Which observation features share most info with affect"""
        print("\n── Mutual Information ──")
        
        mi_valence = mutual_info_regression(X, valence)
        mi_intensity = mutual_info_regression(X, intensity)

        top_val = np.argsort(mi_valence)[::-1][:5]
        top_aro = np.argsort(mi_intensity)[::-1][:5]

        print(f"Top 5 obs features → Valence : {top_val}  scores: {mi_valence[top_val].round(3)}")
        print(f"Top 5 obs features → intensity : {top_aro}  scores: {mi_intensity[top_aro].round(3)}")

    def _feature_importance(self, X, valence, intensity):
        """Random forest importance as cross-check"""
        print("\n── Feature Importance (Random Forest) ──")
        
        rf_val = RandomForestRegressor(n_estimators=50).fit(X, valence)
        rf_aro = RandomForestRegressor(n_estimators=50).fit(X, intensity)

        top_val = np.argsort(rf_val.feature_importances_)[::-1][:5]
        top_aro = np.argsort(rf_aro.feature_importances_)[::-1][:5]

        print(f"Top 5 obs features → Valence : {top_val}")
        print(f"Top 5 obs features → intensity : {top_aro}")