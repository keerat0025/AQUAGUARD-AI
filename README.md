# AquaGuard AI — Water Quality Prediction & Monitoring

A beginner-friendly Streamlit project demonstrating a Random Forest classifier for water-potability dataset labels.

## Features
- Styled dashboard and sidebar navigation
- Nine water-quality input fields
- Dataset upload and model training inside the app
- Real evaluation metrics saved after training
- Screening notes and project information page

## Run locally
Run these commands from the folder containing `app.py`:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Dataset
Use the Kaggle Water Potability dataset CSV named `water_potability.csv`. It must contain:
`ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability`

Upload it through the app or place it at `data/water_potability.csv`.

## Publish on GitHub

Create a GitHub repository, then run these commands from this project folder (replace the URL with your repository URL):

```bash
git init
git add .
git commit -m "Prepare AquaGuard AI for deployment"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

The trained model and dataset are included in the repository. Python bytecode caches, virtual environments, and Streamlit secrets are excluded.

## Deploy on Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize access to the repository.
2. Select **Create app**, then choose the repository, `main` branch, and `app.py` as the main file path.
3. Select **Deploy**. Streamlit installs the packages listed in `requirements.txt`.

The app can also retrain its model from the uploaded dataset. Files written during a hosted session are not persistent across all redeployments or restarts, so keep important dataset or model changes in the repository if they need to persist.

## Limitation
This project is for educational screening only. A model prediction is not a laboratory test or certification that water is safe to drink.
