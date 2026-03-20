# QA Copilot v2 — 3 Phases avec validation humaine

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Processus
- **Phase 1** : Soumettez l'US → l'IA pose des questions → vous répondez → bouton "Valider Phase 1"
- **Phase 2** : L'IA génère le plan de test → vous modifiez si besoin → bouton "Valider Phase 2"
- **Phase 3** : L'IA génère les cas détaillés → export Markdown ou Texte

## Déploiement Streamlit Cloud
1. Pushez sur GitHub
2. Allez sur https://streamlit.io/cloud
3. Connectez le repo → Deploy
