# Objective
The etl pipeline created using python, and pandas for ingesting and transforming data from input csv to output csv.


## Requirements:
- Python version >=3.11


## Setup Steps:
1. Change the working directory to etl using `cd etl`
2. Unzip the file `input_data.zip`
3. Create folder `output_data` in the current folder
3. Create virtual environment using `python -m venv v_env`
4. In windows activate virtual environment using `v_env/scripts/activate` and in linux use `source v_env/bin/activate` to activate
5. Install dependencies using `pip install -r requirements.txt`


## Running the Project:
1. Setup the virtual environment and install dependencies by following above `Setup Steps` and also activate the virtual environment
2. Now run the application using `python main.py` and once the execution completes successfully, a new output file should appear in `output_data/` folder
