call "\Path\To\Anaconda3\Scripts\activate.bat"

call conda activate imgClassifierTrainerWikidata

cd Path\To\ # (main.py's folder parent)

flask --app main.py --debug run
