from flask import Flask, render_template, url_for, request, redirect
from SPARQLWrapper import SPARQLWrapper, JSON
from tensorflow.keras import layers
import tensorflow as tf
from PIL import Image
import requests
import os
import io

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
    """

    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()

    pictures = []

    for result in results["results"]["bindings"]:
        pictures.append(result["pic"]["value"])

    return pictures

def create_model(NUM_CLASSES):
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
    return render_template("home.html")

@app.route("/model_creation", methods=["post"])
def model_creation():
    if request.method == "POST":
        classes_text = request.form["class"]

        CLASSES = classes_text.split()
        NUM_CLASSES = len(CLASSES)

        os.mkdir(os.path.join("C:/Users/brase/PycharmProjects/imgClassifierTrainerWikidata/", "dataset"))

        for CLASS in CLASSES:
            NAME_CLASS = get_name_Q_item(CLASS)

            os.mkdir(os.path.join("C:/Users/brase/PycharmProjects/imgClassifierTrainerWikidata/dataset/", NAME_CLASS))

            imgs = get_item_picture(CLASS)

            i = 1
            for img in imgs:
                imgIO = requests.get(img, stream=True, headers={"User-Agent": "Mozilla/5.0 (Windows Phone 10.0; Android 6.0.1; Microsoft; RM-1152) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Mobile Safari/537.36 Edge/15.15254"}).content
                imgIO = io.BytesIO(imgIO)
                imgIO.seek(0)
                image = Image.open(imgIO)
                image = image.resize((192, 192))
                image.save(f"./dataset/{NAME_CLASS}/{i}.png", 'png')
                i += 1

        batch_size = 16
        img_height = 192
        img_width = 192

        train_ds = tf.keras.utils.image_dataset_from_directory(
            "C:/Users/brase/PycharmProjects/imgClassifierTrainerWikidata/dataset/",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            "C:/Users/brase/PycharmProjects/imgClassifierTrainerWikidata/dataset/",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        model = create_model(NUM_CLASSES)

        model.fit(train_ds, validation_data=val_ds, epochs=200)

        model.save("model.h5")

        return redirect(url_for("home"))
    else:
        return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
