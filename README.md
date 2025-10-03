# xoxoplanet-nasa2025

NASA Space Apps Challenge 2025 with team XOXOplanet.
Challenge: A World Away: Hunting for Exoplanets with AI.

This repository is where we keep all our code, notebooks, and notes. The idea is that we all work from the same shared base so changes can be tracked, merged, and improved together.

## Getting Started

Follow these steps to set up the project on your own computer. Even if you are new to coding or version control, just go step by step and you’ll be ready.

Preferred IDE: VS Code
We recommend using Visual Studio Code. It makes it easier for us to collaborate since we can share workspace settings, use the same extensions, and follow the same workflow.

1. Clone this repository

You need to get a local copy of this repository (the code lives on GitHub, but you’ll want it on your own machine). You have two options:

Option A (easiest): Download as ZIP from GitHub and unzip it into a folder of your choice.

Option B (recommended): Use Git directly.

Open VS Code.

Open the workspace/folder where you want the project to live.

Open the terminal in VS Code by pressing Ctrl + J (Windows) or Cmd + J (Mac).

Type:

git clone https://github.com/matildafroste/xoxoplanet-nasa2025.git


Now you have the project locally on your computer.

2. Install Python

Make sure you have Python 3 installed. You can check by typing:

python --version


If you don’t have it, download it from python.org
.

3. Create a virtual environment

A virtual environment is like a clean sandbox that keeps our project’s libraries separate from other projects. In the project folder, run:

python -m venv venv

4. Activate the virtual environment

This tells your computer to “step into” the sandbox.

On Windows:

venv\Scripts\activate


On Mac/Linux:

source venv/bin/activate


When activated, your terminal should show (venv) at the beginning of the line.

5. Install required libraries

Now install all the libraries we need for the project. They are listed in a file called requirements.txt. Run:

pip install -r requirements.txt


That’s it — you’re ready to run code and start contributing.

## How to Work with GitHub

This is the basic workflow we’ll use so that we can all collaborate smoothly.

1. Pull the latest changes

Before starting work, always update your local copy so you’re working on the latest version:

git pull

2. Make changes locally

Do your coding in VS Code in the project folder. Save your changes.

3. Stage and commit your changes

When you’re happy with your edits, stage them and write a short message explaining what you did:

git add .
git commit -m "Short message about what you changed"

4. Push your changes to GitHub

Send your changes to the shared repository:

git push

5. Communicate

If you’re adding something major (like a new library), let the team know so we all update accordingly.

Notes for the Team

Use VS Code so we work in the same environment.

Always activate your virtual environment before working on the project.

Always pull before starting to code, so you don’t overwrite someone else’s work.

If you add a new library, update requirements.txt so we all stay in sync.

If something doesn’t work, don’t hesitate to ask in the chat so we solve it quickly together.