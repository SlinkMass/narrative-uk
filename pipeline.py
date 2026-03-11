import sys
from news import get_stories
from analysis import run_analysis

def main():
    print("STARTING NARRATIVE UK PIPELINE")
    
    # 1. RSS Fetching & Clustering
    try:
        get_stories(force_refresh=True)
    except Exception as e:
        print(f"Critical Error in news.py: {e}")
        sys.exit(1)
        
    # 2. AI Auditing
    try:
        run_analysis()
    except Exception as e:
        print(f"Critical Error in analysis.py: {e}")
        sys.exit(1)

    print("PIPELINE COMPLETE")

if __name__ == "__main__":
    main()