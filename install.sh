
#!/bin/bash
set -e

echo "================================"
echo " RL Donkey Project Installer"
echo "================================"

# ---- Check python ----
command -v python3 >/dev/null || { echo "Python3 not installed"; exit 1; }

# ---- Create venv ----
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

pip install --upgrade pip

# ---- Install dependencies ----
pip install -r requirements.txt

# ---- Install local repos ----
echo "Installing gym-donkeycar..."
cd gym-donkeycar
pip install -e .
cd ..

echo "Installing rl-baselines3-zoo..."
cd rl-baselines3-zoo
pip install -e .
cd ..

echo "Installing aae-train-donkeycar..."
cd aae-train-donkeycar
pip install -e .
cd ..

echo "================================"
echo " Installation done successfully"
echo "================================"
echo "Activate with: source venv/bin/activate"
