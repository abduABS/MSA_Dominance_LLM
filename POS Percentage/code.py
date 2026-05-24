import csv

def remove_starting_words(input_filepath, output_prefix):
    # The percentages of words to remove
    percentages = [20, 40, 60]
    
    # Read the original CSV data
    try:
        with open(input_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            data = list(reader)
    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        return

    if not data:
        print("Error: The CSV file is empty.")
        return

    # Generate a new file for each percentage
    for percent in percentages:
        output_filename = f"{output_prefix}_{percent}percent.csv"
        
        with open(output_filename, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            
            for row in data:
                new_row = []
                for col_index, cell in enumerate(row):
                    # Keep the first column (Column 1 / Index 0) exactly as is
                    if col_index == 0:
                        new_row.append(cell)
                    else:
                        # Split the cell into words, calculate the cutoff, and rejoin
                        words = cell.split()
                        words_to_remove = int(len(words) * (percent / 100))
                        modified_sentence = " ".join(words[words_to_remove:])
                        new_row.append(modified_sentence)
                        
                writer.writerow(new_row)
                
        print(f"Successfully created: {output_filename}")

# --- How to run the code ---
# Replace 'data.csv' with the name of your actual file
# 'output' will be the prefix for your new files (e.g., output_20percent.csv)

if __name__ == "__main__":
    remove_starting_words('MADAR.combined.Arabic.csv', 'output')