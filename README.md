# wikidata-image-classifier

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Presentation**

A tool to create custom image classifier from wikidata image with tensorflow.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Installation**

*Step 1 :*

Create a new environment with conda or venv in python 3.9.

*Step 2 :*

Clone the repositery in your project's file

```bash
git clone https://github.com/ThorkildFregi/wikidata-image-classifier/
```

*Step 3 :*

Install the dependencies :

```bash
pip install dependencies.txt
```

*Step 4 :*

Change the start file you need :

- On Windows : change, either ```start_conda.bat``` or ```start_venv.bat```, with the instruction in it.
- On Linux : change, either ```start_conda.sh``` or ```start_venv.sh```, with the instruction in it.

*Step 5 :*

Run your start file.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Utilisation**

After running the start file, go to http://127.0.0.1/5000. You'll have a page with a box, in it you can write Q-Items from wikidata. Then, click on the "Create model" button. After some minutes, you will have a ``model.h5``, this is your model. You can load it in any python file. If you need to make an other model, delete the dataset file and go to the website again.
