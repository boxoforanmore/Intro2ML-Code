from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

app = Flask(__name__)

# Form class with simple validators
class HelloForm(Form):
    sayhello = TextAreaField('', [validators.DataRequired()])

# Main page method
@app.route('/')
def index():
    form = HellowForm(request.form)
    return render_template('first_app.html')

# Hello page method
@app.route('/hello', methods=['POST'])
def hello():
    form = HelloForm(request.form)
    if (request.method == 'POST') and (form.validate()):
        name = request.form['sayhello']
        return render_template('hello.html', name=name)
    return render_template('first_app.html', form=form)


if __name__ == '__main__':
    app.run()
