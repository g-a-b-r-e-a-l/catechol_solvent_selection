import pandas as pd
import glob
import os

RESULTS_DIR = "results/decoder"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "all_results.csv")

def main():
    files = glob.glob(os.path.join(RESULTS_DIR, "run_*.csv"))
    if not files:
        print("No result files found.")
        return
    
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Collated {len(files)} files into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
