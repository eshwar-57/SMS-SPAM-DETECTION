from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer
with open("C:\\Users\\eshwa\\Downloads\\sms_spam_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("C:\\Users\\eshwa\\Downloads\\vectorizer.pkl", 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the spam input page
@app.route('/spam')
def spam():
    return render_template('spam.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the message from the form
        message = request.form['message']
        
        # Vectorize the input message using the trained vectorizer
        message_counts = vectorizer.transform([message])
        
        # Make a prediction
        prediction = model.predict(message_counts)
        
        # Determine if it's spam or not
        result = "SPAM" if prediction[0] == 1 else "NOT SPAM"
        
        # Render the result page
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
