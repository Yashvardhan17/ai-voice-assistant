"""Train reproducible baseline classifiers using synthetic telecom queries."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
'billing': ['Why is my bill so high', 'Show me my latest invoice', 'I was charged twice', 'How much do I owe', 'My payment was not applied', 'I need a receipt for my payment', 'Dispute an incorrect charge', 'Check my account balance', 'Why did my monthly charge increase', 'Download my billing statement', 'Refund the extra amount charged', 'Tell me my payment due date'],
'connectivity': ['My internet is not working', 'The connection keeps dropping', 'I cannot connect to wifi', 'My broadband is very slow', 'Web pages will not load', 'No internet connection at my house', 'My network is unstable', 'The internet disconnects constantly', 'My speed is far too low', 'Wifi connects but has no internet', 'I cannot browse the web', 'Internet stopped after restart'],
'account_access': ['I forgot my password', 'Unlock my customer account', 'Cannot sign in to my profile', 'Reset my login credentials', 'My account is locked', 'Help me change my password', 'I forgot my username', 'I cannot access my account', 'Update my registered email', 'I am unable to log in', 'Recover my online account', 'Fix my account authentication'],
'plan_information': ['Show available internet plans', 'Can I upgrade my subscription', 'What does the premium plan include', 'Compare monthly internet packages', 'Find a cheaper broadband plan', 'What is the price of unlimited data', 'Show available speed tiers', 'I want to downgrade my package', 'Which plan supports faster speeds', 'Explain my current subscription', 'Change to a family plan', 'Tell me the plan benefits'],
'service_outage': ['Is there an outage in my area', 'Check for a service disruption', 'When will the network outage end', 'Is the provider down today', 'Are there active network incidents', 'Show outage status for my area', 'Is scheduled maintenance happening nearby', 'Why is the entire neighborhood offline', 'When will services be restored', 'Check active outages', 'Is there a local service interruption', 'The whole area has lost service'],
'technical_support': ['Help me configure my router', 'How do I restart the modem', 'Router shows a blinking red light', 'Guide me through network troubleshooting', 'How do I update router firmware', 'Help with router settings', 'How do I reset the wifi router', 'The modem indicator keeps flashing', 'What cables should I check', 'Assist with device setup', 'How to change wifi channel', 'I need technical troubleshooting steps']
}

def train():
    rows=[{'text':t, 'intent':label} for label, texts in SAMPLES.items() for t in texts]
    df=pd.DataFrame(rows).dropna().drop_duplicates(subset=['text'])
    (ROOT/'data').mkdir(exist_ok=True)
    df.to_csv(ROOT/'data'/'intents.csv',index=False)
    X_train,X_test,y_train,y_test=train_test_split(df.text,df.intent,test_size=0.25,stratify=df.intent,random_state=42)
    results={}; trained={}
    for name,estimator in {'logistic_regression':LogisticRegression(max_iter=1500,random_state=42),'linear_svm':LinearSVC(random_state=42)}.items():
        pipeline=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),('classifier',estimator)])
        pipeline.fit(X_train,y_train); predicted=pipeline.predict(X_test)
        results[name]={'accuracy':float(accuracy_score(y_test,predicted)),'weighted_f1':float(f1_score(y_test,predicted,average='weighted')),'report':classification_report(y_test,predicted,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y_test,predicted,labels=sorted(df.intent.unique())).tolist()}
        trained[name]=pipeline
        print(name, 'accuracy=',round(results[name]['accuracy'],3),'weighted_f1=',round(results[name]['weighted_f1'],3))
    winner=max(results,key=lambda n:results[n]['weighted_f1'])
    # Logistic regression is persisted as well for calibrated probability/confidence.
    (ROOT/'artifacts').mkdir(exist_ok=True)
    joblib.dump(trained['logistic_regression'],ROOT/'artifacts'/'intent_model.joblib')
    joblib.dump(trained[winner],ROOT/'artifacts'/'best_model.joblib')
    features=TfidfVectorizer(stop_words='english').fit_transform(df.text)
    clustering=KMeans(n_clusters=6,random_state=42,n_init=10).fit(features)
    results['clustering']={'num_clusters':6,'cluster_sizes':pd.Series(clustering.labels_).value_counts().sort_index().tolist(),'inertia':float(clustering.inertia_)}
    results['selected_by_weighted_f1']=winner
    results['test_samples']=len(X_test)
    (ROOT/'artifacts'/'metrics.json').write_text(json.dumps(results,indent=2))
    try:
        import mlflow
        mlflow.set_tracking_uri((ROOT/'mlruns').as_uri())
        mlflow.set_experiment('telecom-intent-classification')
        for name in ['logistic_regression','linear_svm']:
            with mlflow.start_run(run_name=name):
                mlflow.log_param('vectorizer','tfidf_1_2gram')
                mlflow.log_metric('test_accuracy',results[name]['accuracy'])
                mlflow.log_metric('test_weighted_f1',results[name]['weighted_f1'])
                mlflow.log_artifact(str(ROOT/'data'/'intents.csv'))
    except ImportError:
        print('MLflow not installed: metrics saved locally in artifacts/metrics.json')
    print('Saved model and actual evaluation metrics to artifacts/')

if __name__=='__main__': train()
