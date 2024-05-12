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
pip install requirements.txt
```

*Step 4 :*

Run ``main.py``.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Utilisation**

After running ``main.py``, go to http://127.0.0.1/5000. You'll have a page with a box, in it you can write Q-Items from wikidata. Then, click on the "Create model" button. After some minutes, you will have a ``model.h5``, this is your model. You can download it and load it in any python file or use it in the application with the predict function.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Test it**

If you want to test it got to <a href="https://huggingface.co/spaces/ThorkildFregi/Wikidata-image-classifier"> this space </a>
