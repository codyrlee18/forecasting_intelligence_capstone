import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sn

def churn_analysis():

    data = pd.read_csv("/opt/airflow/clean_data/churn_model_data.csv")
    data = data[data['deal_status'] == 'Closed -Won']
    join_data = data[['account_id', 'account_name', 'churn']]


    # KDA 

    data = pd.get_dummies(data, columns=['feature_purchased', 'industry', 'region'], drop_first=True)
    data['csat_missing'] = data['csat_score'].isna().astype(int)
    data['comment_sentiment_missing'] = data['comment_sentiment_score'].isna().astype(int)

    X = data[['client_tenure_months', 'sessions_last_30d', 'csat_score', 'comment_sentiment_score', 'support_tickets_30d', 'last_login_days'] + [c for c in data.columns if c.startswith('industry') or c.startswith('feature_purchased') or c.startswith('region')]]
    y = data['churn']

    X['csat_score'].fillna(X['csat_score'].mean(), inplace=True)
    X['comment_sentiment_score'].fillna(X['comment_sentiment_score'].mean(), inplace=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, join_data_train, join_data_test = train_test_split(X, y, join_data, test_size=0.3, random_state=42)


    # Train a Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Get feature importances
    importances = rf_model.feature_importances_

    # Create a Series for better readability
    feature_names = X.columns
    feature_importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    formatted_feature_importance = feature_importance_series.reset_index()
    formatted_feature_importance.columns = ['Feature', 'Importance Score']
    output_file_path = '/opt/airflow/outputs/KDA_output.csv'
    formatted_feature_importance.to_csv(output_file_path, index=False)
    print(f"KDA results saved to {output_file_path}")

    top_5_features = formatted_feature_importance.nlargest(5, 'Importance Score')

    #renaming variables
    pretty_names = {
    'comment_sentiment_score': 'Sentiment Score',
    'csat_score': 'CSAT Score',
    'client_tenure_months': 'Client Tenure (mo)',
    'sessions_last_30d': 'Sessions (30d)',
    'last_login_days': 'Days Since Login',
    'support_tickets_30d': 'Support Tickets (30d)',
    
    'region_LATAM': 'Region: LATAM',
    'region_North America': 'Region: North America',
    'region_Europe': 'Region: Europe',
    
    'feature_purchased_Core': 'Feature: Core',
    'feature_purchased_Automation': 'Feature: Automation',
    'feature_purchased_Analytics': 'Feature: Analytics',
    'feature_purchased_Premium Support': 'Feature: Premium Support',
    
    'industry_EdTech': 'Industry: EdTech',
    'industry_MarTech': 'Industry: MarTech',
    'industry_E -Commerce': 'Industry: E-Commerce',
    'industry_HealthTech': 'Industry: HealthTech',
    'industry_SMB SaaS': 'Industry: SMB SaaS',
    'industry_Logistics': 'Industry: Logistics',
    'industry_FinTech': 'Industry: FinTech',

    'comment_sentiment_missing': 'Missing Sentiment',
    'csat_missing': 'Missing CSAT'
    }

    top_5_features['Feature'] = top_5_features['Feature'].map(pretty_names).fillna(top_5_features['Feature'])

    ax = sn.barplot(x ='Feature', y= 'Importance Score', 
               data= top_5_features,
               color='skyblue',
               orient='v')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout() 
    plt.savefig("/opt/airflow/outputs/kda_plot.png", transparent= True)


    # Generate churn probabilities
    churn_probabilities = rf_model.predict_proba(X_test)

    # The second column (index 1) contains the probability of churn (class 1)
    probability_of_churn = churn_probabilities[:, 1]
    join_data_test["churn_probability"] = probability_of_churn

    high_risk = join_data_test[(join_data_test["churn"] == 0) & (join_data_test["churn_probability"] > 0.7)]
    output_file_path_risk = '/opt/airflow/outputs/high_risk_customers.csv'
    high_risk.to_csv(output_file_path_risk, index=False)
    print(f"Probabilities results saved to {output_file_path}")
        
    # Evaluate the model's performance (e.g., using AUC-ROC)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC-ROC Score: {auc_roc:.2f}")

if __name__ == "__main__":
    churn_analysis()