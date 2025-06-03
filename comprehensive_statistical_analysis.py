import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, ks_2samp, mannwhitneyu
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.api import VAR
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class ComprehensiveAnalysis:
    def __init__(self, data_dir="./data/cleaned_spatial_monthly_data"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("./analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of analysis
        (self.results_dir / "statistical_tests").mkdir(exist_ok=True)
        (self.results_dir / "correlation_analysis").mkdir(exist_ok=True)
        (self.results_dir / "regression_models").mkdir(exist_ok=True)
        (self.results_dir / "econometric_analysis").mkdir(exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)
        
    def load_and_merge_data(self):
        """Load crime data and merge with socioeconomic data"""
        print("Loading and merging datasets...")
        
        # Load crime data (sample from recent months for efficiency)
        crime_files = sorted(list(self.data_dir.glob("2024-*.csv")))[:6]  # Last 6 months of 2024
        
        crime_data = []
        for file in crime_files:
            df = pd.read_csv(file)
            df['Month'] = pd.to_datetime(file.stem.replace('_cleaned_spatial', ''))
            crime_data.append(df)
        
        self.crime_df = pd.concat(crime_data, ignore_index=True)
        
        # Load socioeconomic data
        socio_path = self.data_dir / "Societal_wellbeing_dataset" / "final_merged_cleaned_lsoa_london_social_dataset.csv"
        if socio_path.exists():
            self.socio_df = pd.read_csv(socio_path, encoding='latin1', low_memory=False)
        else:
            print("Socioeconomic data not found. Using crime data only.")
            self.socio_df = None
            
        # Load Premier League data
        prem_path = Path("./prem_matches_cleaned.csv")
        if prem_path.exists():
            self.prem_df = pd.read_csv(prem_path)
            self.prem_df['Date'] = pd.to_datetime(self.prem_df['Date'])
        else:
            self.prem_df = None
            
        print(f"Crime data shape: {self.crime_df.shape}")
        if self.socio_df is not None:
            print(f"Socioeconomic data shape: {self.socio_df.shape}")
        if self.prem_df is not None:
            print(f"Premier League data shape: {self.prem_df.shape}")
            
    def preprocess_data(self):
        """Clean and preprocess the data for analysis"""
        print("Preprocessing data...")
        
        # Clean crime data
        self.crime_df['Month'] = pd.to_datetime(self.crime_df['Month'])
        self.crime_df['Year'] = self.crime_df['Month'].dt.year
        self.crime_df['MonthNum'] = self.crime_df['Month'].dt.month
        self.crime_df['Weekday'] = self.crime_df['Month'].dt.day_name()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['Crime type', 'Last outcome category', 'LSOA name', 'Police Force']
        
        for col in categorical_cols:
            if col in self.crime_df.columns:
                self.crime_df[f'{col}_encoded'] = le.fit_transform(self.crime_df[col].astype(str))
        
        # Create aggregated features
        self.create_aggregated_features()
        
    def create_aggregated_features(self):
        """Create aggregated features for analysis"""
        print("Creating aggregated features...")
        
        # Monthly crime counts by LSOA
        self.monthly_lsoa = self.crime_df.groupby(['LSOA code', 'Month']).agg({
            'Crime ID': 'count',
            'Crime type': 'nunique',
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()
        
        self.monthly_lsoa.columns = ['LSOA_code', 'Month', 'Crime_count', 'Crime_types', 'Avg_lat', 'Avg_lon']
        
        # Monthly crime counts by area
        self.monthly_area = self.crime_df.groupby(['Police Force', 'Month']).agg({
            'Crime ID': 'count',
            'Crime type': 'nunique'
        }).reset_index()
        
        self.monthly_area.columns = ['Police_Force', 'Month', 'Crime_count', 'Crime_types']
        
        # Crime type distribution
        self.crime_type_dist = self.crime_df.groupby(['Crime type', 'Month']).size().reset_index()
        self.crime_type_dist.columns = ['Crime_type', 'Month', 'Count']
        
    def perform_statistical_tests(self):
        """Perform comprehensive statistical tests"""
        print("Performing statistical tests...")
        
        results = {}
        
        # 1. Normality tests
        crime_counts = self.monthly_lsoa['Crime_count']
        
        # Shapiro-Wilk test (for smaller samples)
        if len(crime_counts) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(crime_counts)
            results['shapiro_wilk'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(crime_counts, 'norm')
        results['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_p}
        
        # Anderson-Darling test
        ad_result = stats.anderson(crime_counts, dist='norm')
        results['anderson_darling'] = {
            'statistic': ad_result.statistic,
            'critical_values': ad_result.critical_values,
            'significance_levels': ad_result.significance_levels
        }
        
        # 2. Distribution comparison tests
        # Compare crime counts across different police forces
        forces = self.crime_df['Police Force'].unique()
        if len(forces) >= 2:
            force1_crimes = self.crime_df[self.crime_df['Police Force'] == forces[0]]['Crime ID'].count()
            force2_crimes = self.crime_df[self.crime_df['Police Force'] == forces[1]]['Crime ID'].count()
            
            # Mann-Whitney U test
            mw_stat, mw_p = stats.mannwhitneyu(
                self.monthly_area[self.monthly_area['Police_Force'] == forces[0]]['Crime_count'],
                self.monthly_area[self.monthly_area['Police_Force'] == forces[1]]['Crime_count'],
                alternative='two-sided'
            )
            results['mann_whitney'] = {'statistic': mw_stat, 'p_value': mw_p}
        
        # 3. Seasonal analysis
        monthly_crimes = self.crime_df.groupby(self.crime_df['Month'].dt.month)['Crime ID'].count()
        
        # Kruskal-Wallis test for seasonal differences
        month_groups = [self.crime_df[self.crime_df['Month'].dt.month == month]['Crime ID'].count() 
                       for month in range(1, 13) if month in self.crime_df['Month'].dt.month.unique()]
        
        if len(month_groups) > 2:
            kw_stat, kw_p = stats.kruskal(*month_groups)
            results['kruskal_wallis_seasonal'] = {'statistic': kw_stat, 'p_value': kw_p}
        
        # 4. Time series stationarity test
        adf_result = adfuller(self.monthly_lsoa.groupby('Month')['Crime_count'].sum())
        results['augmented_dickey_fuller'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4]
        }
        
        # Save results
        self.save_statistical_results(results)
        return results
    
    def correlation_analysis(self):
        """Perform comprehensive correlation analysis"""
        print("Performing correlation analysis...")
        
        # Prepare numeric data for correlation
        numeric_data = self.monthly_lsoa[['Crime_count', 'Crime_types', 'Avg_lat', 'Avg_lon']].copy()
        
        # Add time-based features
        numeric_data['Month_num'] = self.monthly_lsoa['Month'].dt.month
        numeric_data['Year'] = self.monthly_lsoa['Month'].dt.year
        
        # Pearson correlation
        pearson_corr = numeric_data.corr(method='pearson')
        
        # Spearman correlation
        spearman_corr = numeric_data.corr(method='spearman')
        
        # Kendall correlation
        kendall_corr = numeric_data.corr(method='kendall')
        
        # Partial correlation (controlling for time trend)
        from scipy.stats import pearsonr
        partial_corr = {}
        for col1 in numeric_data.columns:
            for col2 in numeric_data.columns:
                if col1 != col2:
                    # Control for year trend
                    x = numeric_data[col1]
                    y = numeric_data[col2]
                    z = numeric_data['Year']
                    
                    # Residuals after regressing on control variable
                    x_res = sm.OLS(x, sm.add_constant(z)).fit().resid
                    y_res = sm.OLS(y, sm.add_constant(z)).fit().resid
                    
                    partial_corr[f"{col1}_vs_{col2}"] = pearsonr(x_res, y_res)[0]
        
        # Visualize correlations
        self.plot_correlation_matrices(pearson_corr, spearman_corr, kendall_corr)
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr,
            'partial': partial_corr
        }
    
    def regression_analysis(self):
        """Perform multiple regression analyses"""
        print("Performing regression analysis...")
        
        # Prepare data
        X = self.monthly_lsoa[['Crime_types', 'Avg_lat', 'Avg_lon']].copy()
        X['Month_num'] = self.monthly_lsoa['Month'].dt.month
        X['Year'] = self.monthly_lsoa['Month'].dt.year
        
        y = self.monthly_lsoa['Crime_count']
        
        # Remove any NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {}
        
        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        models['linear_regression'] = {
            'model': lr,
            'r2_score': r2_score(y_test, y_pred_lr),
            'mse': mean_squared_error(y_test, y_pred_lr),
            'mae': mean_absolute_error(y_test, y_pred_lr)
        }
        
        # 2. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        
        models['ridge_regression'] = {
            'model': ridge,
            'r2_score': r2_score(y_test, y_pred_ridge),
            'mse': mean_squared_error(y_test, y_pred_ridge),
            'mae': mean_absolute_error(y_test, y_pred_ridge)
        }
        
        # 3. Lasso Regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        
        models['lasso_regression'] = {
            'model': lasso,
            'r2_score': r2_score(y_test, y_pred_lasso),
            'mse': mean_squared_error(y_test, y_pred_lasso),
            'mae': mean_absolute_error(y_test, y_pred_lasso)
        }
        
        # 4. Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        models['random_forest'] = {
            'model': rf,
            'r2_score': r2_score(y_test, y_pred_rf),
            'mse': mean_squared_error(y_test, y_pred_rf),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'feature_importance': dict(zip(X.columns, rf.feature_importances_))
        }
        
        # 5. Statsmodels OLS for detailed statistics
        X_sm = sm.add_constant(X)
        ols_model = sm.OLS(y, X_sm).fit()
        
        models['ols_detailed'] = {
            'model': ols_model,
            'summary': ols_model.summary(),
            'r2': ols_model.rsquared,
            'adj_r2': ols_model.rsquared_adj,
            'f_statistic': ols_model.fvalue,
            'f_pvalue': ols_model.f_pvalue
        }
        
        # Diagnostic tests
        residuals = ols_model.resid
        fitted = ols_model.fittedvalues
        
        # Heteroskedasticity tests
        bp_test = het_breuschpagan(residuals, X_sm)
        white_test = het_white(residuals, X_sm)
        
        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(residuals)
        
        models['diagnostics'] = {
            'breusch_pagan': {'statistic': bp_test[0], 'p_value': bp_test[1]},
            'white_test': {'statistic': white_test[0], 'p_value': white_test[1]},
            'durbin_watson': dw_stat
        }
        
        self.plot_regression_diagnostics(models, X_test, y_test)
        
        return models
    
    def econometric_analysis(self):
        """Perform econometric analysis including time series and panel data models"""
        print("Performing econometric analysis...")
        
        results = {}
        
        # 1. Time series analysis
        ts_data = self.monthly_lsoa.groupby('Month').agg({
            'Crime_count': 'sum',
            'Crime_types': 'mean'
        }).reset_index()
        
        ts_data = ts_data.sort_values('Month')
        ts_data.set_index('Month', inplace=True)
        
        # Vector Autoregression (VAR)
        if len(ts_data) > 10:  # Need sufficient observations
            try:
                var_data = ts_data[['Crime_count', 'Crime_types']].dropna()
                
                # Determine optimal lag order
                max_lags = min(4, len(var_data) // 4)
                if max_lags > 0:
                    var_model = VAR(var_data)
                    lag_order = var_model.select_order(maxlags=max_lags)
                    
                    # Fit VAR model
                    optimal_lag = min(lag_order.aic, lag_order.bic, 2)  # Cap at 2 lags
                    var_fitted = var_model.fit(optimal_lag)
                    
                    results['var_model'] = {
                        'lag_order': optimal_lag,
                        'aic': var_fitted.aic,
                        'bic': var_fitted.bic,
                        'summary': var_fitted.summary()
                    }
                    
                    # Granger causality test
                    if len(var_data) > optimal_lag + 1:
                        try:
                            granger_crime_to_types = grangercausalitytests(
                                var_data[['Crime_types', 'Crime_count']], 
                                maxlag=optimal_lag, 
                                verbose=False
                            )
                            results['granger_causality'] = {
                                'crime_to_types': granger_crime_to_types
                            }
                        except:
                            print("Granger causality test failed")
            except Exception as e:
                print(f"VAR analysis failed: {e}")
        
        # 2. Panel data analysis (if we have LSOA-level data)
        if 'LSOA_code' in self.monthly_lsoa.columns:
            panel_data = self.monthly_lsoa.copy()
            panel_data['Month_num'] = panel_data['Month'].dt.month
            panel_data['Year'] = panel_data['Month'].dt.year
            
            # Fixed effects regression
            try:
                # Simple fixed effects model using dummy variables for LSOA
                lsoa_dummies = pd.get_dummies(panel_data['LSOA_code'], prefix='LSOA')
                X_panel = pd.concat([
                    panel_data[['Crime_types', 'Month_num', 'Year']],
                    lsoa_dummies
                ], axis=1)
                
                y_panel = panel_data['Crime_count']
                
                # Remove NaN values
                mask = ~(X_panel.isna().any(axis=1) | y_panel.isna())
                X_panel = X_panel[mask]
                y_panel = y_panel[mask]
                
                if len(X_panel) > 0:
                    X_panel_sm = sm.add_constant(X_panel)
                    fe_model = sm.OLS(y_panel, X_panel_sm).fit()
                    
                    results['fixed_effects'] = {
                        'r2': fe_model.rsquared,
                        'adj_r2': fe_model.rsquared_adj,
                        'f_statistic': fe_model.fvalue,
                        'f_pvalue': fe_model.f_pvalue
                    }
            except Exception as e:
                print(f"Fixed effects model failed: {e}")
        
        return results
    
    def premier_league_analysis(self):
        """Analyze relationship between Premier League matches and crime"""
        if self.prem_df is None:
            print("Premier League data not available")
            return None
            
        print("Analyzing Premier League impact on crime...")
        
        results = {}
        
        # London teams
        london_teams = ['Arsenal', 'Chelsea', 'Tottenham', 'West Ham']
        london_matches = self.prem_df[
            (self.prem_df['Home'].isin(london_teams)) | 
            (self.prem_df['Away'].isin(london_teams))
        ].copy()
        
        # Match day analysis
        london_matches['Match_month'] = london_matches['Date'].dt.to_period('M')
        match_counts = london_matches.groupby('Match_month').size().reset_index()
        match_counts.columns = ['Month', 'Match_count']
        match_counts['Month'] = match_counts['Month'].dt.to_timestamp()
        
        # Merge with crime data
        crime_monthly = self.crime_df.groupby(self.crime_df['Month'].dt.to_period('M')).size().reset_index()
        crime_monthly.columns = ['Month', 'Crime_count']
        crime_monthly['Month'] = crime_monthly['Month'].dt.to_timestamp()
        
        merged_data = pd.merge(crime_monthly, match_counts, on='Month', how='left')
        merged_data['Match_count'] = merged_data['Match_count'].fillna(0)
        
        # Correlation analysis
        if len(merged_data) > 5:
            correlation = merged_data[['Crime_count', 'Match_count']].corr().iloc[0, 1]
            
            # Statistical test
            stat, p_value = pearsonr(merged_data['Crime_count'], merged_data['Match_count'])
            
            results['correlation_with_matches'] = {
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Match outcome analysis
        london_home_matches = london_matches[london_matches['Home'].isin(london_teams)]
        
        if len(london_home_matches) > 0:
            # Win/Loss analysis
            london_home_matches['Result'] = london_home_matches.apply(
                lambda row: 'Win' if row['Home Goals'] > row['Away Goals'] 
                else ('Loss' if row['Home Goals'] < row['Away Goals'] else 'Draw'), 
                axis=1
            )
            
            result_counts = london_home_matches['Result'].value_counts()
            results['match_outcomes'] = result_counts.to_dict()
        
        return results
    
    def save_statistical_results(self, results):
        """Save statistical test results to file"""
        output_file = self.results_dir / "statistical_tests" / "test_results.txt"
        
        with open(output_file, 'w') as f:
            f.write("COMPREHENSIVE STATISTICAL ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for test_name, test_results in results.items():
                f.write(f"{test_name.upper().replace('_', ' ')}\n")
                f.write("-" * 30 + "\n")
                
                if isinstance(test_results, dict):
                    for key, value in test_results.items():
                        f.write(f"{key}: {value}\n")
                else:
                    f.write(f"Result: {test_results}\n")
                
                f.write("\n")
        
        print(f"Statistical results saved to {output_file}")
    
    def plot_correlation_matrices(self, pearson, spearman, kendall):
        """Plot correlation matrices"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Pearson correlation
        sns.heatmap(pearson, annot=True, cmap='coolwarm', center=0, ax=axes[0])
        axes[0].set_title('Pearson Correlation')
        
        # Spearman correlation
        sns.heatmap(spearman, annot=True, cmap='coolwarm', center=0, ax=axes[1])
        axes[1].set_title('Spearman Correlation')
        
        # Kendall correlation
        sns.heatmap(kendall, annot=True, cmap='coolwarm', center=0, ax=axes[2])
        axes[2].set_title('Kendall Correlation')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "correlation_analysis" / "correlation_matrices.png", dpi=300)
        plt.close()
    
    def plot_regression_diagnostics(self, models, X_test, y_test):
        """Plot regression diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Model comparison
        model_names = ['Linear', 'Ridge', 'Lasso', 'Random Forest']
        r2_scores = [models[key]['r2_score'] for key in ['linear_regression', 'ridge_regression', 'lasso_regression', 'random_forest']]
        
        axes[0, 0].bar(model_names, r2_scores)
        axes[0, 0].set_title('Model Comparison (R² Score)')
        axes[0, 0].set_ylabel('R² Score')
        
        # Feature importance (Random Forest)
        if 'feature_importance' in models['random_forest']:
            features = list(models['random_forest']['feature_importance'].keys())
            importance = list(models['random_forest']['feature_importance'].values())
            
            axes[0, 1].barh(features, importance)
            axes[0, 1].set_title('Feature Importance (Random Forest)')
            axes[0, 1].set_xlabel('Importance')
        
        # Residual plot (OLS)
        if 'ols_detailed' in models:
            residuals = models['ols_detailed']['model'].resid
            fitted = models['ols_detailed']['model'].fittedvalues
            
            axes[1, 0].scatter(fitted, residuals, alpha=0.6)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_title('Residuals vs Fitted')
            axes[1, 0].set_xlabel('Fitted Values')
            axes[1, 0].set_ylabel('Residuals')
        
        # Q-Q plot
        if 'ols_detailed' in models:
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "regression_models" / "regression_diagnostics.png", dpi=300)
        plt.close()
    
    def generate_report(self, statistical_results, correlation_results, regression_results, econometric_results, prem_results):
        """Generate comprehensive analysis report"""
        report_file = self.results_dir / "comprehensive_analysis_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Statistical and Econometric Analysis Report\n\n")
            
            f.write("## Executive Summary\n")
            f.write("This report presents a comprehensive statistical and econometric analysis of crime data, ")
            f.write("incorporating socioeconomic factors and sports event impacts.\n\n")
            
            f.write("## Data Overview\n")
            f.write(f"- Crime records analyzed: {len(self.crime_df):,}\n")
            f.write(f"- Time period: {self.crime_df['Month'].min()} to {self.crime_df['Month'].max()}\n")
            f.write(f"- Unique LSOAs: {self.crime_df['LSOA code'].nunique():,}\n\n")
            
            f.write("## Statistical Test Results\n")
            f.write("### Normality Tests\n")
            if 'kolmogorov_smirnov' in statistical_results:
                ks_result = statistical_results['kolmogorov_smirnov']
                f.write(f"- Kolmogorov-Smirnov: statistic={ks_result['statistic']:.4f}, p-value={ks_result['p_value']:.4f}\n")
            
            f.write("### Stationarity Test\n")
            if 'augmented_dickey_fuller' in statistical_results:
                adf_result = statistical_results['augmented_dickey_fuller']
                f.write(f"- Augmented Dickey-Fuller: statistic={adf_result['statistic']:.4f}, p-value={adf_result['p_value']:.4f}\n")
            
            f.write("\n## Correlation Analysis\n")
            if correlation_results:
                pearson_crime_types = correlation_results['pearson'].loc['Crime_count', 'Crime_types']
                f.write(f"- Crime count vs Crime types (Pearson): {pearson_crime_types:.4f}\n")
            
            f.write("\n## Regression Analysis\n")
            f.write("### Model Performance Comparison\n")
            for model_name, results in regression_results.items():
                if 'r2_score' in results:
                    f.write(f"- {model_name.replace('_', ' ').title()}: R² = {results['r2_score']:.4f}\n")
            
            if econometric_results and 'var_model' in econometric_results:
                f.write("\n## Econometric Analysis\n")
                f.write("### Vector Autoregression (VAR)\n")
                var_results = econometric_results['var_model']
                f.write(f"- Optimal lag order: {var_results['lag_order']}\n")
                f.write(f"- AIC: {var_results['aic']:.4f}\n")
                f.write(f"- BIC: {var_results['bic']:.4f}\n")
            
            if prem_results and 'correlation_with_matches' in prem_results:
                f.write("\n## Sports Events Impact\n")
                f.write("### Premier League Matches and Crime\n")
                corr_results = prem_results['correlation_with_matches']
                f.write(f"- Correlation: {corr_results['correlation']:.4f}\n")
                f.write(f"- Significant: {corr_results['significant']}\n")
            
            f.write("\n## Recommendations\n")
            f.write("1. Continue monitoring temporal patterns in crime data\n")
            f.write("2. Consider seasonal adjustments in resource allocation\n")
            f.write("3. Investigate spatial clustering patterns\n")
            f.write("4. Examine impact of major events on crime patterns\n")
        
        print(f"Comprehensive report saved to {report_file}")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive statistical and econometric analysis...")
        
        # Load and preprocess data
        self.load_and_merge_data()
        self.preprocess_data()
        
        # Perform analyses
        statistical_results = self.perform_statistical_tests()
        correlation_results = self.correlation_analysis()
        regression_results = self.regression_analysis()
        econometric_results = self.econometric_analysis()
        prem_results = self.premier_league_analysis()
        
        # Generate report
        self.generate_report(
            statistical_results, 
            correlation_results, 
            regression_results, 
            econometric_results, 
            prem_results
        )
        
        print("Analysis completed successfully!")
        print(f"Results saved in: {self.results_dir}")
        
        return {
            'statistical': statistical_results,
            'correlation': correlation_results,
            'regression': regression_results,
            'econometric': econometric_results,
            'premier_league': prem_results
        }

if __name__ == "__main__":
    # Run the analysis
    analyzer = ComprehensiveAnalysis()
    results = analyzer.run_full_analysis() 