"""
Module: predict.py
CRISP-DM Phase 6: Deployment Inference Utility
Loads the serialized Scikit-Learn pipeline and generates profit predictions for new startup data.
"""

import os
import sys
import argparse
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "startup_profit_pipeline.pkl")

def load_inference_pipeline(model_path=MODEL_PATH):
    """Loads the pre-trained scikit-learn pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Please run 'python main.py' first."
        )
    return joblib.load(model_path)

def predict_single(pipeline, rd_spend, admin_spend, marketing_spend, state):
    """Predicts profit for a single startup profile."""
    input_df = pd.DataFrame([{
        "R&D Spend": float(rd_spend),
        "Administration": float(admin_spend),
        "Marketing Spend": float(marketing_spend),
        "State": str(state)
    }])
    pred = pipeline.predict(input_df)[0]
    return pred

def predict_batch(pipeline, csv_path, output_path=None):
    """Predicts profit for a batch CSV file and saves or returns results."""
    df = pd.read_csv(csv_path)
    required_cols = ["R&D Spend", "Administration", "Marketing Spend", "State"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in input CSV.")
    
    preds = pipeline.predict(df[required_cols])
    df["Predicted_Profit"] = preds.round(2)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[Batch Prediction] Results saved to: {output_path}")
    return df

def interactive_mode(pipeline):
    """Prompts the user interactively in the terminal."""
    print("\n" + "=" * 60)
    print("50 STARTUPS PROFIT PREDICTOR (CRISP-DM DEPLOYMENT INTERFACE)")
    print("=" * 60)
    try:
        rd = float(input("Enter R&D Spend ($) [e.g. 120000]: ").strip() or "120000")
        admin = float(input("Enter Administration Spend ($) [e.g. 100000]: ").strip() or "100000")
        mkt = float(input("Enter Marketing Spend ($) [e.g. 250000]: ").strip() or "250000")
        print("Available States: California, New York, Florida")
        state = input("Enter State [Default: California]: ").strip() or "California"
        
        profit = predict_single(pipeline, rd, admin, mkt, state)
        print("\n" + "-" * 60)
        print("INPUT SUMMARY:")
        print(f"  R&D Spend:       ${rd:,.2f}")
        print(f"  Administration:  ${admin:,.2f}")
        print(f"  Marketing Spend: ${mkt:,.2f}")
        print(f"  State:           {state}")
        print("-" * 60)
        print(f"ESTIMATED PROFIT:  ${profit:,.2f}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Error during input processing: {e}")

def main():
    parser = argparse.ArgumentParser(description="Predict startup profit using trained scikit-learn model.")
    parser.add_argument("--rd", type=float, help="R&D Expenditure in USD")
    parser.add_argument("--admin", type=float, help="Administration Expenditure in USD")
    parser.add_argument("--marketing", type=float, help="Marketing Expenditure in USD")
    parser.add_argument("--state", type=str, choices=["California", "New York", "Florida"], help="Operational State")
    parser.add_argument("--batch", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--output", type=str, help="Path to output CSV for batch prediction results")

    args = parser.parse_args()
    pipeline = load_inference_pipeline()

    if args.batch:
        df_res = predict_batch(pipeline, args.batch, args.output)
        print(df_res[["R&D Spend", "Administration", "Marketing Spend", "State", "Predicted_Profit"]].head())
    elif args.rd is not None and args.admin is not None and args.marketing is not None and args.state is not None:
        profit = predict_single(pipeline, args.rd, args.admin, args.marketing, args.state)
        print(f"Predicted Profit: ${profit:,.2f}")
    else:
        interactive_mode(pipeline)

if __name__ == "__main__":
    main()
