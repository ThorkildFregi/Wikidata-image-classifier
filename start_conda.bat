call "\Path\To\Anaconda3\Scripts\activate.bat"

call conda activate imgClassifierTrainerWikidata

cd Path\To\(parents folder of main.py)

flask --app main.py --debug run
