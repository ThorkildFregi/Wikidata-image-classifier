from flask import Flask, render_template, url_for, request, redirect, send_from_directory
from SPARQLWrapper import SPARQLWrapper, JSON
from tensorflow.keras import layers
import tensorflow as tf
import multiprocessing
from PIL import Image, ImageFile
import urllib.request
import numpy as np
import requests
import shutil
import os
import io

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

def get_name_Q_item(Q_ITEM):
    endpoint_url = "https://query.wikidata.org/sparql"
    user_agent = "Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; RM-1152) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Mobile Safari/537.36 Edge/15.15254"
    query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
        PREFIX wd: <http://www.wikidata.org/entity/> 
        SELECT * WHERE {
                wd:""" + Q_ITEM + """ rdfs:label ?label .
                FILTER (langMatches( lang(?label), "EN" ) )
        } 
        LIMIT 1
        """

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        final_result = result["label"]["value"]

    return final_result

def get_item_picture(item):
    endpoint_url = "https://query.wikidata.org/sparql"
    user_agent = "Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; RM-1152) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Mobile Safari/537.36 Edge/15.15254"
    query = """
    SELECT ?item ?itemLabel ?pic WHERE {
      ?item wdt:P31 wd:""" + item + """;
        wdt:P18 ?pic.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 100
    """

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()

    pictures = []

    for result in results["results"]["bindings"]:
        pictures.append(result["pic"]["value"])

    return pictures

def create_image_classifier(NUM_CLASSES):
    model = tf.keras.Sequential([
        layers.Input(shape=[192, 192, 3]),
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(3136),
        layers.Dense(128),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(optimizer="adam", loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

    return model

@app.route("/")
def home():
    if "model.h5" not in os.listdir():
        return redirect(url_for("create_model"))

    return render_template("home.html")

@app.route('/loading-prediction', methods=["post"])
def loading_prediction():
    if request.method == "POST":
        file = request.files["image"]

        image = Image.open(file)
        image.save("prediction_image.png", "png")

        return render_template("loading.html", next_page="prediction")
    else:
        return redirect(url_for("home"))

@app.route('/prediction', methods=["get"])
def prediction():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "./dataset/",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(192, 192),
        batch_size=16)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "./dataset/",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(192, 192),
        batch_size=16)

    class_names = train_ds.class_names

    model = tf.keras.models.load_model("model.h5")

    image = Image.open("prediction_image.png")

    image = image.resize((192, 192))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    percentage = round(100 * np.max(score), 2)

    os.remove("prediction_image.png")

    return render_template("result.html", pclass=predicted_class, percentage=percentage)

@app.route('/create_model')
def create_model():
    return render_template("create_model.html")

@app.route('/loading-model-creation', methods=["post"])
def loading_model_creation():
    if request.method == "POST":
        classes_text = request.form["class"]
        epochs = request.form["epochs"]

        return render_template("loading.html", next_page="model_creation", classes=classes_text, epochs=epochs)
    else:
        return redirect(url_for("home"))

@app.route("/model-creation", methods=["post"])
def model_creation():
    if request.method == "POST":
        if "dataset" in os.listdir():
            shutil.rmtree("./dataset/")

        if "model.h5" in os.listdir():
            os.remove("model.h5")

        classes_text = request.form["class"]
        epochs = request.form["epochs"]

        CLASSES = classes_text.split()

        os.mkdir(os.path.join(os.path.dirname(__file__), "dataset"))

        for CLASS in CLASSES:
            NAME_CLASS = get_name_Q_item(CLASS)

            os.mkdir(os.path.join("./dataset/", NAME_CLASS))

            imgs = get_item_picture(CLASS)
            
            i = 0
            for img in imgs:
                response = requests.get(img, stream=True, headers={"User-Agent": "Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; RM-1152) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Mobile Safari/537.36 Edge/15.15254"})
                image = Image.open(io.BytesIO(response.content))
                image = image.resize((192, 192))
                image.save(f"./dataset/{NAME_CLASS}/{i}.png", 'png')
                i += 1

        batch_size = 16
        img_height = 192
        img_width = 192

        train_ds = tf.keras.utils.image_dataset_from_directory(
            "./dataset/",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            "./dataset/",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        NUM_CLASSES = len(train_ds.class_names)

        model = create_image_classifier(NUM_CLASSES)

        model.fit(train_ds, validation_data=val_ds, epochs=int(epochs))

        model.save("model.h5")

        return redirect(url_for("home"))
    else:
        return redirect(url_for("home"))


@app.route('/uploads/model.h5')
def download_model():
    path = os.path.dirname(__file__)

    return send_from_directory(path, "model.h5")

if __name__ == "__main__":
    app.run(debug=True)
