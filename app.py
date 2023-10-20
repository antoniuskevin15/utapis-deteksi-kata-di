from flask import Flask, render_template, request, jsonify, send_from_directory
from Models.DiWordDetector import DiWordDetector

app = Flask(__name__, template_folder='Pages')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_diword():
    if request.method == 'POST':
        paragraph = request.form.get('paragraph')
        detector = DiWordDetector()
        result = detector.detect_di_usage(paragraph)
        return render_template('result.html', result=result, paragraph=paragraph)

@app.route('/detect-di-word', methods=['POST'])
def detect_diword_api():
    if request.method == 'POST':
        try:
            # Get the JSON data from the request
            data = request.get_json()

            paragraph = data.get('paragraph')

            detector = DiWordDetector()
            result = detector.detect_di_usage(paragraph)

            finalResult = {
                "result": result,
                "paragraph": paragraph
            }

            return jsonify(finalResult)

        except Exception as e:
            return jsonify({'error': str(e)})

        

if __name__ == '__main__':
    app.run()