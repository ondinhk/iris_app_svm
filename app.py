from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html", Predicted_flower_name="Predicted flower name", url_image='static/setosa.jpg')


@app.route('/', methods=['POST'])
def output():
    sl = request.form["slengthin"]
    sw = request.form["swidthin"]
    pl = request.form["plengthin"]
    pw = request.form["pwidthin"]
    input = [[sl, sw, pl, pw]]
    # Gọi model lên
    loaded_model = pickle.load(open('model_svm', 'rb'))
    # Dự đoán mô hình
    result = loaded_model.predict(input)
    # Lấy tên
    name = result[0]
    # Lấy url image
    if name == "Versicolor":
        url_image = 'static/setosa.jpg'
    elif name == "Virginica":
        url_image = 'static/versicolor.jpg'
    else:
        url_image = 'static/virginica.jpg'

    return render_template("index.html", Predicted_flower_name=name, url_image=url_image)


if __name__ == '__main__':
    app.run(debug=True)

# 1 1 2 3