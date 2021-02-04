
import argparse
from CALINet import TrainPredict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CALI-Net. Hyperameters file can be found in experiment setup.")
    parser.add_argument('--start_week',type=int, default='9',help='Region for prediction')
    parser.add_argument('--end_week',type=int, default='16',help='Region for prediction')
    args = parser.parse_args()
    start_week = args.start_week
    end_week = args.end_week
    for epiweek in range(start_week,end_week+1): 
        currentWeek=epiweek+32  # convert
        TrainPredict(currentWeek,epiweek) 
