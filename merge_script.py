import subprocess
import sys

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return decode_output(output), decode_output(error), process.returncode

def decode_output(byte_string):
    encodings = ['utf-8', 'latin-1', 'ascii']
    for encoding in encodings:
        try:
            return byte_string.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_string.decode('utf-8', errors='replace')

def check_and_merge():
    print("Fetching changes from origin...")
    output, error, code = run_command("git fetch origin")
    if code != 0:
        print(f"Error fetching from origin:\n{error}")
        return

    print("Checking current branch...")
    current_branch, error, code = run_command("git rev-parse --abbrev-ref HEAD")
    if code != 0:
        print(f"Error getting current branch:\n{error}")
        return
    current_branch = current_branch.strip()

    print(f"Current branch: {current_branch}")
    
    print("Checking for potential conflicts...")
    merge_base_cmd = f"git merge-base HEAD origin/{current_branch}"
    merge_base, error, code = run_command(merge_base_cmd)
    if code != 0:
        print(f"Error finding merge base:\n{error}")
        return
    merge_base = merge_base.strip()

    merge_tree_cmd = f"git merge-tree {merge_base} HEAD origin/{current_branch}"
    output, error, code = run_command(merge_tree_cmd)
    
    if '<<<<<<<' in output:
        print("Potential conflicts detected. Merge aborted.")
        print("Here are the potential conflicts:")
        print(output)
        return

    print("No conflicts detected. Proceeding with merge...")
    output, error, code = run_command(f"git merge origin/{current_branch}")
    
    if code != 0:
        print(f"Error during merge:\n{error}")
    else:
        print("Merge successful!")
        print(output)

if __name__ == "__main__":
    check_and_merge()
