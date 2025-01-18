# Project Setup Instructions for Python Development on GitHub

Welcome to the project! This guide will help you set up everything needed to start coding and contributing effectively. Follow these steps carefully, and feel free to ask for help if you get stuck.

---

## 1. Install Visual Studio Code (VS Code)

### Steps:
1. Go to the [Visual Studio Code website](https://code.visualstudio.com/).
2. Download and install the latest version of VS Code for your operating system.
3. Launch VS Code and install the following required extensions:
   - **Python** (by Microsoft): Provides support for Python code editing, debugging, and IntelliSense.
   - **Remote - WSL** (by Microsoft): Enables integration between VS Code and the WSL environment for seamless coding.

---

## 2. Install Windows Subsystem for Linux (WSL)

### Steps:
1. Open PowerShell as Administrator.
2. Run the following command to install WSL and the latest Ubuntu version:
   ```bash
   wsl --install
   ```
3. Restart your computer after the installation is complete.
4. If the `wsl --install` command does not work, ensure WSL features are enabled manually:
   - Open PowerShell as Administrator and run:
     ```bash
     dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
     dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
     ```
   - Restart your computer.
   - After restarting, run the `wsl --install` command again.
5. Open the Ubuntu terminal (search for "Ubuntu" in your start menu).
6. Set up your username and password when prompted.
7. Update your Ubuntu packages:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

---

## 3. Create a Folder for the Project

### Steps:
1. Decide where you want to host the project on your computer.
2. Create a dedicated folder for the repository using the file explorer (GUI) or the terminal:
   - **Using the file explorer (recommended):**
     - Open your file manager.
     - Navigate to the desired location (e.g., Documents or Desktop).
     - Right-click and select **New Folder**. Name it something descriptive, like `MyProject`.
   - **Using the terminal:**
     ```bash
     mkdir /path/to/your/folder
     ```

---

## 4. Open the Terminal in VS Code

### Steps:
1. Open Visual Studio Code.
2. Use the **Remote - WSL** extension to open the project:
   - Press `Ctrl+Shift+P`.
   - Type "Remote-WSL: Open Folder" and select it.
   - Navigate to the folder you created earlier and open it.
3. Open the integrated terminal:
   - Press `` Ctrl+` `` (backtick).
   - Ensure the terminal displays "Ubuntu" and **not PowerShell** in the dropdown at the top-right.
   - If "Ubuntu" is not selected, click the dropdown and choose "Ubuntu" from the list.

---

## 5. Clone the GitHub Repository

### Steps:
1. In the Ubuntu terminal within VS Code, check your current directory:
   ```bash
   ls
   ```
   Ensure you're in the correct folder where you want to host the project.
2. If needed, navigate to the folder you created for the project:
   ```bash
   cd /path/to/your/folder
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/JuanPabloRoldan/HyPyRider.git # Use this URL directly, no quotes or brackets needed
   ```
4. Navigate into the project folder:
   ```bash
   cd HyPyRider
   ```

---

## 6. Set Up SSH for GitHub

### Steps:
1. Generate an SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com" # Replace your_email@example.com with your actual email, without the quotes
   ```
   - Press Enter to accept the default file location.
   - If prompted, set a passphrase or press Enter to skip.

2. Start the SSH agent and add your key:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. Copy your SSH key to the clipboard:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   - Copy the entire output.

4. Add the key to your GitHub account:
   - Go to [GitHub SSH settings](https://github.com/settings/keys).
   - Click **New SSH Key**.
   - Paste the key into the "Key" field and give it a descriptive title.
   - Click **Add SSH Key**.

5. Test your connection:
   ```bash
   ssh -T git@github.com
   ```
   You should see a success message confirming your SSH setup.

6. Update the repository remote URL to use SSH:
   ```bash
   git remote set-url origin git@github.com:JuanPabloRoldan/HyPyRider.git # Use this URL directly, no quotes or brackets needed
   ```

Now you can use `git pull` and `git push` without entering your username and password.

---

## 7. Install Python and Required Packages

### Steps:
1. Check if Python is installed:
   ```bash
   python3 --version
   ```
   If Python is not installed, install it:
   ```bash
   sudo apt install python3 python3-pip -y
   ```
2. Install the project dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

---

## 8. Start Coding!

### Steps:
1. Open the project folder in VS Code:
   - Use the **File** menu and select **Open Folder**.
   - Navigate to your project folder and open it.

2. Open any file you want to edit by clicking on it in the file explorer (left sidebar).
   - **Important:** Avoid editing directly on the `main` branch.

3. Always check your branch before starting work:
   ```bash
   git branch
   ```
   If you're on `main`, switch to a different branch:
   ```bash
   git checkout -b your-feature-branch
   ```

4. Pull the latest changes from the remote repository before making any edits:
   ```bash
   git pull origin main
   ```

5. Navigate to the correct folder in the terminal to run the Python script:
   ```bash
   cd /path/to/your/folder/HyPyRider
   ```
   Then execute the script:
   ```bash
   python3 main.py
   ```

6. Alternatively, you can run the script directly in VS Code:
   - Open the file (e.g., `main.py`) in the editor.
   - Click the green "Run" button in the top-right corner of the editor, or press `F5` (ensure Python is set as the interpreter).

---

### Common Commands Cheat Sheet (Linux):
- **Navigation:**
  - `pwd` : Print the current working directory.
  - `ls` : List files and directories.
  - `cd directory_name` : Change directory.
  - `cd ..` : Move up one directory.
- **File Operations:**
  - `touch filename` : Create an empty file.
  - `mkdir directory_name` : Create a new directory.
  - `rm filename` : Remove a file.
  - `rm -r directory_name` : Remove a directory and its contents.
- **Git Commands:**
  - `git status` : Check the status of your repository.
  - `git pull origin main` : Pull the latest changes from the main branch.
  - `git add .` : Stage all changes.
  - `git commit -m "Your commit message"` : Commit changes with a message.
  - `git push origin main` : Push changes to the main branch.

---

## 9. Need Help?
If you're unsure about a command or encounter any issues, use ChatGPT:
1. Describe your problem or the command you're trying to use.
2. Ask ChatGPT for help directly in the integrated terminal or browser.

### Example:
"How do I create a virtual environment for Python?"
ChatGPT will provide a step-by-step answer.

---

By following this guide, you should be ready to start contributing to the project. Happy coding!