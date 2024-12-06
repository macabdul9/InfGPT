import os
import pandas as pd
import argparse

def calculate_accuracy(ground_truth, prediction):
    """
    Calculate accuracy by comparing ground truth and predictions.
    """
    return round((ground_truth == prediction).mean() * 100, 2)

def process_model_directory(root_dir, model_dir):
    """
    Process all CSV files in the model directory to calculate task accuracies and save results.
    """
    tasks = []
    accuracies = []
    
    for file_name in os.listdir(model_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(model_dir, file_name)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Ensure the necessary columns are present
                if 'ground_truth' in df.columns and 'prediction' in df.columns:
                    accuracy = calculate_accuracy(df['ground_truth'], df['prediction'])
                    task_name = os.path.splitext(file_name)[0]  # Extract task name from file name
                    tasks.append(task_name)
                    accuracies.append(accuracy)
                else:
                    print(f"Skipping {file_name}: Missing 'ground_truth' or 'prediction' columns")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Save the results for the model
    if tasks:
        # Add average accuracy
        average_accuracy = round(sum(accuracies) / len(accuracies), 2)
        tasks.append("Average")
        accuracies.append(average_accuracy)

        # Save results to a CSV file
        results_df = pd.DataFrame({'Task': tasks, 'Accuracy': accuracies})
        model_name = os.path.basename(model_dir)
        output_path = os.path.join(root_dir, f"{model_name}.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print(f"No valid CSV files found in {model_dir}")

def main(root_dir):
    """
    Main function to process all model directories in the root directory.
    """
    for model_name in os.listdir(root_dir):
        model_dir = os.path.join(root_dir, model_name)
        if os.path.isdir(model_dir):
            print(f"Processing model: {model_name}")
            process_model_directory(root_dir=root_dir, model_dir=model_dir)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process models to calculate task accuracies.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory containing model directories.")
    args = parser.parse_args()
    
    # Call the main function
    main(args.root_dir)
