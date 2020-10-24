This folder contains the files from a Heteroskedasticity lesson in Dr. Goenners Econ 534 class from the fall 2017 semester. The lesson was orignally done in STATA. The lesson is repeated in Python. The original STATA.do file is included.
The Python version is done twice.
1st with loading files directly from STATA using pd.read_stata.
The 2nd version is reading files directly from Oracle creating an engine and using pd.read_sql from sqlalchemy.
The excercise is available for download as a Jupyter Notebook file, a python file, and is also reproduced using R.
To recreate lesson in Python with a .dta(STATA) file
 1. Load Anaconda with the desired versions of R and Python (optional) https://repo.anaconda.com/archive/
 2. If using python directly, skip step 1 and load the .py file directly into python.
 3. Put .dta file in your desired location
 4. Load the Jupyter notebook 534-Heteroskedasticity.ipynb file.
 5. Edit the 534-Heteroskedasticity.ipynb file  or .py to change file locations
 6. Execute
 
To recreate lesson in Python with Oracle file
 1. Load Anaconda with the desired versions of R and Python (optional) https://repo.anaconda.com/archive/
 2. Put .csv files in desired location
 3. Run SQL in 534-HethomePriceCreateTables.txt to create tables in your Oracle database (db password is in file)
 4. Edit the Load_homepricev1.txt for file locations and save as a .ctl file
 5. Edit the Load_hprice1.txt file for file locations and save as a .ctl file
 6. Edit and run the sqlloader program for each file from your command prompt to import the data
 6a. --C:\Oracle>sqlldr C##HET C:\Users\jfras\OneDrive\UND\534AppliedEcon\Datasets\Load_homepricev10.ctl
 --Password
 6b. --C:\Oracle>sqlldr C##HET C:\Users\jfras\OneDrive\UND\534AppliedEcon\Datasets\Load_hprice1.ctl
 --Password:
 7. Load the 534-Heteroskedasticity-OracleConnect.ipynb
 8. Edit the 534-Heteroskedasticity-OracleConnect.ipynb file to change file locations
 9. Execute
