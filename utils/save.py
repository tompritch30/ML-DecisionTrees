import pickle
from joblib import dump

def save_model_pkl(model, filepath):
    """Saves the trained model to a file.
    
    Args:
        model: The trained model instance to save.
        filepath (str): The path to the file where the model will be saved.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)

        from joblib import dump

def save_model_jl(model, filename):
    """
    Saves the model using joblib.
    """
    dump(model, filename)

def load_model_pkl(filepath):
    """Loads a trained model from a file.
    
    Args:
        filepath (str): The path to the file from which the model will be loaded.
    
    Returns:
        The loaded model instance.
    """
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model
def process_data(input_file, output_file):
    # Open the input file to read data
    with open(input_file, 'r') as f:
        # Assuming the data is separated by new lines for each row
        rows = f.readlines()

    processed_data = []

    for row in rows:
        # Strip to remove any leading/trailing whitespaces including newline characters,
        # and then split each row into columns based on commas
        columns = row.strip().split(',')
        
        # Check if the row contains the expected number of columns before processing
        if len(columns) != 17:
            print("Error: Row does not contain exactly 17 columns.")
            continue
        
        # Move the first column to the end
        new_row = columns[1:] + [columns[0]]
        
        # Join the modified row with commas and add it to the list
        processed_data.append(','.join(new_row))

    # Open the output file to write the processed data
    with open(output_file, 'w') as f:
        # Join each row with a newline character to write into the file
        f.write('\n'.join(processed_data))

# You can then call this function with your specific input and output file paths
# process_data('path_to_input_file.csv', 'path_to_output_file.csv')

def filter_data_by_letter(input_file, output_file, letters):
    # Open the input file to read data
    with open(input_file, 'r') as f:
        # Read each line of the file, assuming each row is on a new line
        rows = f.readlines()

    filtered_data = []

    for row in rows:
        # Strip to remove any leading/trailing whitespaces and split into columns
        columns = row.strip().split(',')
        
        # Check if the row's last column matches any of the specified letters
        if columns[-1] in letters:
            # If it matches, add the whole row to the filtered data list
            filtered_data.append(row.strip())  # Keeping strip() to remove potential newline characters

    # Open the output file to write the filtered data
    with open(output_file, 'w') as f:
        # Join each row with a newline character to write into the file
        f.write('\n'.join(filtered_data))


def split_data(input_file, train_file, test_file, test_ratio=0.2):
    import random
    
    # Open the input file to read data
    with open(input_file, 'r') as f:
        # Read all lines from the file, ensuring each line ends with a newline
        rows = [line if line.endswith('\n') else line + '\n' for line in f]

    # Shuffle the rows to ensure random distribution
    random.shuffle(rows)

    # Calculate the split index based on the specified test ratio
    split_index = int(len(rows) * test_ratio)

    # Split the data into test and training sets
    test_data = rows[:split_index]
    train_data = rows[split_index:]

    # Write the test data to the test file, ensuring each line is properly separated
    with open(test_file, 'w') as f:
        f.writelines(test_data)

    # Write the training data to the train file, ensuring each line is properly separated
    with open(train_file, 'w') as f:
        f.writelines(train_data)