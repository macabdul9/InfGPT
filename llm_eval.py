import os
import requests
import pandas as pd
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Initialize normalizer
normalizer = BasicTextNormalizer()

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")  # Default if not specified

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

def calculate_accuracy(data, args):
    """
    Calculate the accuracy of the model output.

    Args:
        data (pd.DataFrame): The data with model output.
        args (argparse.Namespace): Parsed arguments.

    Returns:
        float: The accuracy of the model output.
    """
    # Accuracy = Correct / Correct + Incorrect
    # Calculate the counts of "Correct" and "Incorrect"
    correct_count = data[args.llm_eval_output_column].str.strip().value_counts().get('Correct', 0)
    incorrect_count = data[args.llm_eval_output_column].str.strip().value_counts().get('Incorrect', 0)

    # Compute accuracy
    accuracy = round((correct_count / (correct_count + incorrect_count))*100, 2)
    return accuracy

def eval_example(prompt, temperature):
    """
    Classify if LLM generated output is correct or not .

    Args:
        ground_truth (str): The reference text.
        generated_output (str): The model's generated text.
        prompt_template (str): The prompt template to format the input.
        temperature (float): The temperature parameter for the API.

    Returns:
        str: The classification result from the API.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
    }
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    response_data = response.json()
    return response_data['choices'][0]['message']['content']

def process_csv_with_checkpoint(file_path, prompt_template, args):
    """
    Process a single CSV file for hallucination classification with checkpointing.

    Args:
        file_path (str): Path to the CSV file.
        prompt_template (str): The prompt template for the classification.
        args (argparse.Namespace): Parsed arguments.
    """
    checkpoint_path = f"{file_path}.checkpoint"
    
    if os.path.exists(checkpoint_path):
        # Load from checkpoint if it exists
        data = pd.read_csv(checkpoint_path)
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        # Load the original file
        data = pd.read_csv(file_path)

        # Check if the file has already been processed
        if args.llm_eval_output_column in data.columns:
            print(f"File {file_path} has already been processed. Skipping.")
            return

        data[args.llm_eval_output_column] = None  # Initialize the model output column

    # Process each row
    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc=f"Processing {file_path}"):
        
        if pd.notnull(row[args.llm_eval_output_column]):  # Skip rows that are already processed
            continue

        instruction = row[args.instruction_column]
        options = row[args.options_column]
        answer = row[args.answer_column]
        generated_output = row[args.generated_answer_column]
        
        formatted_prompt = prompt_template.format(
            instruction=instruction,
            options=options,
            ground_truth=answer,
            generated_response=generated_output
        )
        try:
            
            llm_output = eval_example(prompt=formatted_prompt, temperature=args.temperature)
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e} ! Retrying...")
            llm_output = eval_example(prompt=formatted_prompt, temperature=args.temperature)
        except Exception as e:
            print(f"Error occurred again: {e}")
            llm_output = "Error"

        data.at[index, args.llm_eval_output_column] = llm_output

        # Save checkpoint after every `checkpoint_steps` rows
        if (index + 1) % args.checkpoint_steps == 0:
            data.to_csv(checkpoint_path, index=False)
            print(f"Checkpoint saved at: {checkpoint_path}")

    # Save the final file and remove the checkpoint if all rows are processed
    if data[args.llm_eval_output_column].isnull().sum() == 0:
        data.to_csv(file_path, index=False)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        print(f"Processing completed. Updated file saved: {file_path}. Checkpoint removed.")
    else:
        print(f"Processing incomplete. Checkpoint saved at: {checkpoint_path}")
    # Calculate accuracy an save it in the file. 
    
    # Compute accuracy
    accuracy = calculate_accuracy(data, args)
    return accuracy

    

def evaluate_all(prompt_template, args):
    """
    Process all CSV files in the directory with checkpointing.

    Args:
        prompt_template (str): The prompt template for the classification.
        args (argparse.Namespace): Parsed arguments.
    """
    root_dir = args.root_dir
    
    args.llm_eval_output_column = args.llm_eval_output_column or f"{MODEL_NAME}_output"
        
    for model in os.listdir(root_dir):
        
        dirpath = os.path.join(root_dir, model)
        
        tasks = []
        accuracies = []
        
        folder_name = dirpath.split("/")[-1]
        
        filenames = os.listdir(dirpath)
        
        for filename in filenames:
            
            if filename.endswith('.csv'):
                
                task = filename.split(".")[0]
                
                file_path = os.path.join(dirpath, filename)

                # Check if the file already has the output_column
                with open(file_path, "r") as f:
                    first_line = f.readline()
                    if args.llm_eval_output_column in first_line:
                        print(f"File {file_path} already processed. Skipping.")
                        # calculate accuracy
                        data = pd.read_csv(file_path)
                        # Calculate accuracy an save it in the file.
                        accuracy = calculate_accuracy(data, args)
                        tasks.append(task)
                        accuracies.append(accuracy)
                        continue

                print(f"Processing file: {file_path}")
                accuracy = process_csv_with_checkpoint(file_path=file_path, prompt_template=prompt_template, args=args)
                tasks.append(task)
                accuracies.append(accuracy)
        # Save the accuracy in a file
        accuracy_df = pd.DataFrame({"Task": tasks, "Accuracy": accuracies})
        accuracy_df.to_csv(f"{folder_name}_accuracy.csv", index=False)

def main():
    """
    Main function to handle argument parsing and start processing CSV files.
    """
    parser = argparse.ArgumentParser(description='Evaluate LLM outputs saved in CSV files with checkpointing.')
    parser.add_argument('--root_dir', type=str, default="SpeechGenResults/", help='Root directory to search for CSV files.')
    parser.add_argument('--prompt_file', type=str, default="prompts/prompt.txt", help='Path to the prompt template file.')
    parser.add_argument('--instruction_column', type=str, default="instruction", help='Coloum in which input instructions are stored.')
    parser.add_argument("--options_column", type=str, default="options", help="Column in which multiple choice options are stored.")
    parser.add_argument('--answer_column', type=str, default="answer", help='Coloum for correct answer also called ground truth.')
    parser.add_argument('--generated_answer_column', type=str, default="generated_answer", help='Custom column name for hallucination classification results (optional).')
    parser.add_argument('--llm_eval_output_column', type=str, default=None, help='Custom column name for hallucination classification results (optional).')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature parameter for the API.')
    parser.add_argument('--checkpoint_steps', type=int, default=100, help='Number of steps between each checkpoint.')
    args = parser.parse_args()

    # Load the prompt template
    # This is to avoid reading the prompt file for each CSV file
    with open(args.prompt_file, "r") as prompt_file:
        prompt_template = prompt_file.read()

    # Process CSV files with checkpointing
    evaluate_all(prompt_template=prompt_template,args=args)

if __name__ == "__main__":
    main()