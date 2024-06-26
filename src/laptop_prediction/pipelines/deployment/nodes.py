"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 0.19.2
"""
import os
import json
import pickle
import wandb

def compare_with_champion(model_challenger, score_challenger):
    model_path = "data/04_model/best_model.pickle"
    score_path = "data/05_model_output/score_best_model.json"
    
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    
    if os.path.exists(model_path) and os.path.exists(score_path):
        with open(score_path, "r") as f:
            best_model_score = json.load(f)
        
        challenger_mae = score_challenger['val_mae']
        champion_mae = best_model_score['val_mae']
        
        if challenger_mae > champion_mae:
            best_model_score = score_challenger
            best_model = model_challenger
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            with open(score_path, "w") as f:
                json.dump(best_model_score, f)
            message = (f"New challenger model is better. Updated the champion model.\n"
                       f"Challenger val_mae: {challenger_mae}\n"
                       f"Previous champion val_mae: {champion_mae}")
        else:
            with open(model_path, "rb") as f:
                best_model = pickle.load(f)
            message = (f"Champion model is still better. No update made.\n"
                       f"Challenger val_mae: {challenger_mae}\n"
                       f"Champion val_mae: {champion_mae}")
    else:
        best_model_score = score_challenger
        best_model = model_challenger
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        with open(score_path, "w") as f:
            json.dump(best_model_score, f)
        challenger_mae = score_challenger['val_mae']
        message = (f"No existing champion model. Set the challenger as the champion model.\n"
                   f"Challenger val_mae: {challenger_mae}")
    
    wandb.init(project='PJA-ASI-12C-GR4')
    wandb.log({
        "Val MAE": best_model_score['val_mae'],
        "Val MSE": best_model_score['val_mse'],
        "Val RMSE": best_model_score['val_rmse'],
        "Val R2": best_model_score['val_r2'],
    })
    
    print(message)
    return best_model, best_model_score