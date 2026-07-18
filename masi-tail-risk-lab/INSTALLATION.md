# 🛠️ INSTALLATION — Guide anti-blocage (Windows / VS Code)

Ce projet est **complet et fonctionnel** — tout le code (EVT, GARCH-EVT, HMM,
marges, dashboard) est déjà écrit et testé. Les erreurs rencontrées
(`ModuleNotFoundError`, kernel qui ne redémarre pas) sont **100 %
environnementales**, pas des bugs du code. Suis ces étapes dans l'ordre et
ça fonctionnera.

## 1. Ouvrir le BON dossier

Dans VS Code : **File → Open Folder** → sélectionne le dossier
`masi-tail-risk-lab` (celui qui contient `src/`, `app/`, `notebooks/`, pas
un dossier parent comme "Downloads").

## 2. Créer un environnement virtuel propre (fortement recommandé)

Ouvre un terminal dans VS Code (**Terminal → New Terminal**) et tape :

```bash
python -m venv .venv
```

Puis active-le :
- **Windows (PowerShell)** : `.venv\Scripts\Activate.ps1`
- **Windows (CMD)** : `.venv\Scripts\activate.bat`

Tu dois voir `(.venv)` apparaître au début de la ligne du terminal — c'est
la preuve que l'environnement est actif.

⚠️ Si PowerShell refuse d'activer (erreur "execution policy"), lance
d'abord : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` puis
recommence.

## 3. Installer TOUTES les dépendances d'un coup

Toujours dans le terminal, **avec `(.venv)` actif** :

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Cette commande installe `pandas`, `numpy`, `scipy`, `arch`, `hmmlearn`,
`streamlit`, `plotly`, `reportlab`, `jupyterlab`, `pytest` — tout ce dont
le projet a besoin, en une seule fois. Compte 2-5 minutes.

⚠️ Si `hmmlearn` échoue à s'installer (demande un compilateur C++ sous
Windows) : installe "Microsoft C++ Build Tools" (lien donné dans le message
d'erreur), ou en dernier recours commente temporairement la ligne
`hmmlearn` dans `requirements.txt` — seule la section HMM du dashboard sera
indisponible, tout le reste fonctionne sans.

## 4. Sélectionner le BON interpréteur Python dans VS Code

C'est l'étape que presque tout le monde oublie et qui cause
`ModuleNotFoundError` même après un `pip install` réussi :

1. Ouvre la palette de commandes : `Ctrl+Shift+P`
2. Tape "Python: Select Interpreter"
3. Choisis celui qui contient `.venv` dans son chemin
   (ex: `.\.venv\Scripts\python.exe`)

Pour un notebook Jupyter spécifiquement : ouvre le notebook, clique sur le
nom du kernel **en haut à droite** de l'éditeur, et sélectionne le même
environnement `.venv`.

## 5. Redémarrer proprement

- **Notebook** : ferme complètement l'onglet, rouvre-le, puis relance
  toutes les cellules depuis le début (menu **Run → Run All**).
- **Si le kernel refuse de redémarrer** : ferme VS Code entièrement
  (pas juste l'onglet) et rouvre-le.

## 6. Vérifier que tout fonctionne

Dans le terminal (`.venv` actif) :

```bash
python -m pytest tests/ -q
```

Tu dois voir `6 passed`. Si oui, **tout le projet est opérationnel**.

## 7. Lancer le dashboard

```bash
streamlit run app/dashboard.py
```

Un onglet de navigateur s'ouvre sur `http://localhost:8501`.

---

## Erreurs fréquentes et solutions rapides

| Erreur | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'arch'` | Package non installé OU mauvais interpréteur sélectionné | Étapes 3 et 4 ci-dessus |
| Kernel qui ne répond plus | VS Code figé | Fermer/rouvrir VS Code entièrement |
| `hmmlearn` échoue à l'installation | Compilateur C++ manquant (Windows) | Installer "Microsoft C++ Build Tools" ou commenter la ligne dans requirements.txt |
| Le notebook tourne mais utilise l'ancien code après une modif de `src/` | Cache d'import Python | Redémarrer le kernel (pas juste relancer les cellules) |
| `FileNotFoundError: data/raw/masi.csv` | Mauvais dossier de travail | Vérifie que tu as bien ouvert le dossier `masi-tail-risk-lab` (pas un parent) dans VS Code |
