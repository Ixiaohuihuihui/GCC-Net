import os

def find_hidden_files(folder_path):
    hidden_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.startswith('.'):
                hidden_files.append(os.path.join(root, filename))
    return hidden_files

if __name__ == "__main__":
    folder_path = "./"  # Replace with the path to your target folder

    hidden_files = find_hidden_files(folder_path)
    if hidden_files:
        print("Hidden files found:")
        for file in hidden_files:
            print(file)
    else:
        print("No hidden files found.")

