import os

dataset_path = r"D:\alphanumber\validation"

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    if os.path.isdir(folder_path):
        try:
            ascii_value = int(folder_name)
            
            # Digits 0-9
            if 48 <= ascii_value <= 57:
                new_name = chr(ascii_value)  # 0, 1, 2...
            # Uppercase A-Z
            elif 65 <= ascii_value <= 90:
                new_name = "upper_" + chr(ascii_value)  # upper_A, upper_B...
            # Lowercase a-z
            elif 97 <= ascii_value <= 122:
                new_name = "lower_" + chr(ascii_value)  # lower_a, lower_b...
            else:
                continue
            
            new_path = os.path.join(dataset_path, new_name)
            
            if os.path.exists(new_path):
                print(f"Skipped: {folder_name} -> {new_name} (already exists)")
            else:
                os.rename(folder_path, new_path)
                print(f"Renamed: {folder_name} -> {new_name}")
        except ValueError:
            continue
