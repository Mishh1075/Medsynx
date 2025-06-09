import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.services.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    """
    Preprocess data for synthetic generation.
    
    Args:
        args: Command line arguments
    """
    try:
        # Load data
        logger.info(f"Loading data from {args.input_file}")
        data = pd.read_csv(args.input_file)
        
        # Initialize processor
        processor = DataProcessor()
        
        # Validate data
        logger.info("Validating data...")
        processor.validate_data(data)
        
        # Parse sensitive features
        sensitive_features = args.sensitive_features.split(',') if args.sensitive_features else None
        
        # Preprocess data
        logger.info("Preprocessing data...")
        loader, info = processor.preprocess_data(
            data,
            target_column=args.target_column,
            sensitive_features=sensitive_features
        )
        
        # Save preprocessing info
        info_file = Path(args.output_file).with_suffix('.json')
        pd.Series(info).to_json(info_file)
        logger.info(f"Saved preprocessing info to {info_file}")
        
        # Save preprocessed data
        loader.data.to_csv(args.output_file, index=False)
        logger.info(f"Saved preprocessed data to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for synthetic generation")
    
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save preprocessed CSV file"
    )
    
    parser.add_argument(
        "--target-column",
        type=str,
        help="Name of target column for supervised learning"
    )
    
    parser.add_argument(
        "--sensitive-features",
        type=str,
        help="Comma-separated list of sensitive feature columns"
    )
    
    args = parser.parse_args()
    main(args) 