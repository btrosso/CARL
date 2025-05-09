import os

# === CONFIGURE THESE ===
target_dir = "/Users/brosso/Documents/personal_code/CARL/algotraffic_low_qual/"
append_text = "_r2"  # Text to append before file extension

# === SCRIPT LOGIC ===
for filename in os.listdir(target_dir):
    full_path = os.path.join(target_dir, filename)

    if os.path.isfile(full_path):
        name, ext = os.path.splitext(filename)
        new_name = f"{name}{append_text}{ext}"
        new_full_path = os.path.join(target_dir, new_name)

        os.rename(full_path, new_full_path)
        print(f"Renamed: {filename} → {new_name}")
