from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, render_template, request
from matcher import analyze

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(analyze(data.get("resume", ""), data.get("jd", "")))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        app.logger.exception("analysis failed")
        return jsonify(error="Analysis failed. Please try again in a moment."), 500

if __name__ == "__main__":
    app.run(debug=True)
