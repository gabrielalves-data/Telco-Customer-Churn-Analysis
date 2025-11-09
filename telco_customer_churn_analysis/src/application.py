import os
from flask import Flask, render_template_string, request, redirect, url_for, flash, render_template, session
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from authlib.integrations.flask_client import OAuth
from functools import wraps

from telco_customer_churn_analysis.telco_customer_churn_analysis_app import (
    data_preprocessing_app, exploratory_analysis_app, hypothesis_test_app, train_evaluate_deploy_app,
    predict_with_best_profit_threshold_app, predict_with_xai_app)

dotenv_path = os.path.join(os.path.dirname(__file__), "../../.env")
dotenv_path = os.path.abspath(os.path.normpath(dotenv_path))
load_dotenv(dotenv_path)

application = Flask(__name__)
application.secret_key = os.getenv("FLASK_SECRET_KEY", "flask_key")

oauth = OAuth(application)

oauth.register(
    name='cognito',
    client_id=os.getenv('COGNITO_APP_CLIENT_ID'),
    client_secret=os.getenv('COGNITO_APP_CLIENT_SECRET'),
    server_metadata_url=f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USERPOOL_ID')}/.well-known/openid-configuration",
    client_kwargs={'scope': 'email openid'} 
)


def login_required(f):
    """Decorator to ensure user is logged in via Cognito"""
    @wraps(f)
    def decorator(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorator


def extract_features_from_form(form):
    def conv_int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None
    def conv_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return {
        "City": form.get("City"),
        "Gender": form.get("Gender"),
        "Senior_Citizen": form.get("Senior_Citizen"),
        "Partner": form.get("Partner"),
        "Dependents": form.get("Dependents"),
        "Tenure_Months": conv_int(form.get("Tenure_Months")),
        "Phone_Service": form.get("Phone_Service"),
        "Multiple_Lines": form.get("Multiple_Lines"),
        "Internet_Service": form.get("Internet_Service"),
        "Online_Security": form.get("Online_Security"),
        "Online_Backup": form.get("Online_Backup"),
        "Device_Protection": form.get("Device_Protection"),
        "Tech_Support": form.get("Tech_Support"),
        "Streaming_TV": form.get("Streaming_TV"),
        "Streaming_Movies": form.get("Streaming_Movies"),
        "Contract": form.get("Contract"),
        "Paperless_Billing": form.get("Paperless_Billing"),
        "Payment_Method": form.get("Payment_Method"),
        "Monthly_Charges": conv_float(form.get("Monthly_Charges")),
        "Total_Charges": conv_float(form.get("Total_Charges")),
    }


base_css = """
<style>
body { font-family: Arial, sans-serif; background: #f4f6f8; margin: 0; padding: 0;}
.container { max-width: 800px; margin: 30px auto; background: white; padding: 20px; border-radius: 6px; box-shadow: 0 0 10px #ccc;}
h1, h2 { color: #333; }
input[type=text], input[type=number], select { width: 100%; padding: 8px; margin: 6px 0 12px; border: 1px solid #ccc; border-radius: 4px; }
button { background-color: #007BFF; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
button:hover { background-color: #0056b3; }
.flash { padding: 10px; border-radius: 4px; margin-bottom: 15px;}
.flash-success { background-color: #d4edda; color: #155724; }
.flash-error { background-color: #f8d7da; color: #721c24; }
pre { background: #eee; padding: 15px; border-radius: 6px; overflow-x: auto; }
label { font-weight: bold; }
a { text-decoration: none; color: #007BFF; }
a:hover { text-decoration: underline; }

/* === Loading overlay and spinner === */
#loading-overlay {
  display: none;
  position: fixed;
  z-index: 9999;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background: #f4f6f8;
  backdrop-filter: blur(4px);
  text-align: center;
  align-items: center;
  justify-content: center;
  flex-direction: column;
}

.spinner {
  border: 6px solid #f3f3f3;
  border-top: 6px solid #007BFF;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin-bottom: 15px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

#loading-text {
  color: #007BFF;
  font-size: 1.2em;
  font-weight: bold;
}

#loading-text-time {
  color: #007BFF;
  font-size: 0.9em;
  font-weight: bold;
}
</style>

<script>
function showLoading() {
  document.getElementById('loading-overlay').style.display = 'flex';
}
</script>
"""


@application.route("/")
def index():
    user_logged_in = "user" in session
    print('User: ', user_logged_in)
    return render_template_string(base_css + """
    <style>
      .top-left {
        position: absolute;
        top: 20px;
        right: 20px;
      }
    </style>
                                  
    <div id="loading-overlay">
      <div class="spinner"></div>
      <div id="loading-text">Loading page... please wait</div>
      <div id="loading-text-time">...Can take up to 5 min...</div>
    </div>
                                  
    <div class="top-left">
      {% if user_logged_in %}
        <p>Welcome, {{ session['user']['email'] }}!</p>
        <a href="{{ url_for('logout') }}"><button>Logout</button></a>
      {% else %}
        <a href="{{ url_for('login') }}"><button>Login</button></a>
      {% endif %}
    </div>

    <div class="container">
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          {% for category, message in messages %}
            <div class="flash flash-{{ category }}">{{ message }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}
                                  
      <h1>Telco Customer Churn Analysis</h1>
      <ul>
        <li><a href="{{ url_for('run_eda') }}" onclick="showLoading()">Run Exploratory Data Analysis (EDA)</a></li>
        <li><a href="{{ url_for('run_hypothesis_tests') }}" onclick="showLoading()">Run Hypothesis Tests (Chi2)</a></li>
        <li><a href="{{ url_for('run_train_evaluate_deploy') }}" onclick="showLoading()">Train, Evaluate, Deploy Model</a></li>
        <li><a href="{{ url_for('predict_best_threshold') }}" onclick="showLoading()">Predict with Best Profit Threshold</a></li>
        <li><a href="{{ url_for('predict_xai') }}" onclick="showLoading()">Predict with Explainable AI (XAI)</a></li>
      </ul>
    </div>
    """, user_logged_in=user_logged_in)


@application.route("/login")
def login():
    redirect_uri = url_for('authorize', _external=True)
    return oauth.cognito.authorize_redirect(redirect_uri, prompt='login')


@application.route("/authorize")
def authorize():
    token = oauth.cognito.authorize_access_token()
    user = token['userinfo']
    session['user'] = user
    flash("Logged in successfully!", "success")
    return redirect(url_for('index'))


@application.route("/logout")
def logout():
    session.pop('user', None)
    flash("Logged out successfully", "success")

    return redirect(url_for('index'))


@application.route("/eda")
@login_required
def run_eda():
    try:
        df, preprocessing_text = data_preprocessing_app()
        df, images = exploratory_analysis_app(df)

        return render_template("eda.html", images=images, preprocessing_text=preprocessing_text)
    
    except Exception as e:
        flash(f"EDA failed: {e}", "error")
        
        return redirect(url_for("index"))


@application.route("/hypothesis_tests", methods=["GET", "POST"])
@login_required
def run_hypothesis_tests():
    if request.method == "POST":
        data_choice = request.form.get("data_choice")
        col1 = request.form.get('col1')
        col2 = request.form.get('col2')
        try:
            result_text = hypothesis_test_app(data_choice=data_choice, col1=col1, col2=col2)
            print('Test by data choice')
            
            return render_template("hypothesis_results.html", result=result_text, source=data_choice)
        except Exception as e:
            flash(f"Hypothesis tests failed: {e}", "error")
            print('Error on test')
            
            return redirect(url_for("index"))

    print('Test outside the if')
    return render_template_string(base_css + """
        <style>
          .back-button {
              position: fixed;
              top: 10px;
              left: 10px;
              text-decoration: none;
              font-size: 16px;
              color: #333;
              border: 1px solid #ccc;
              padding: 8px 12px;
              border-radius: 5px;
              background-color: #f9f9f9;
              transition: background-color 0.3s ease;
              z-index: 1000;
          }
          .back-button:hover {
              background-color: #e0e0e0;
          }
        </style>

        <a href="{{ url_for('index') }}" class="back-button" title="Back to Home">
          &#x1F3E0;
        </a>
        <div class="container">
        <h2>Run Hypothesis Tests (Chi2)</h2>
        <form method="POST">
            <label for="data_choice">Choose Data:</label>
            <select id="data_choice" name="data_choice" required>
            <option value="Test">Preprocessed Training Dataset (Test)</option>
            <option value="New">Generate Synthetic Test Data (New)</option>
            </select>
            <br><br>
            <label for="col1">Column 1:</label>
            <input type="text" id="col1" name="col1" placeholder="Enter first column (optional)">
            <br><br>
            <label for="col2">Column 2:</label>
            <input type="text" id="col2" name="col2" placeholder="Enter second column (optional)">
            <br><br>
            <button type="submit">Run Tests</button>
        </form>
        </div>
    """)


@application.route("/train_evaluate_deploy")
@login_required
def run_train_evaluate_deploy():
    try:
        result_text = train_evaluate_deploy_app()

        return render_template("model_results.html", result=result_text)
    except Exception as e:
        flash(f"Train/Evaluate/Deploy failed: {e}", "error")
        return redirect(url_for("index"))


@application.route("/predict_best_threshold", methods=["GET", "POST"])
@login_required
def predict_best_threshold():
    if request.method == "POST":
        features = extract_features_from_form(request.form)
        print('Extracted features: ', features)

        try:
            num_lines = int(request.form.get('num_lines', 100))
            if num_lines <= 0:
                num_lines = 100

        except ValueError:
            num_lines = 100

        try:
            abc_assignment = request.form.get("abc_assignment") == "on"
            print('Try get threshold, and table')
            print('Abc Assign: ', abc_assignment)
            threshold, html_table = predict_with_best_profit_threshold_app(**features, abc_assignment=abc_assignment)
            print('Predicted threshold: ', threshold)

            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')

            if table:
                rows = table.find_all('tr')
                header = rows[0]
                body_rows = rows[1:num_lines + 1]
                new_table = [header] + body_rows
                table.clear()

                for row in new_table:
                    table.append(row)

                html_table = str(table)

            styled_table = f"""
            <style>
              .prediction-table-wrapper {{
                background: #ffffff;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow-x: auto;
                max-width: 100%;
              }}
              table.table {{
                width: 100% !important;
                border-collapse: collapse;
                font-size: 0.9rem;
              }}
              table.table th, table.table td {{
                white-space: nowrap;
                text-align: center;
                padding: 0.5rem 0.75rem;
              }}
              .back-button {{
                position: fixed;
                top: 10px;
                left: 10px;
                text-decoration: none;
                font-size: 16px;
                color: #333;
                border: 1px solid #ccc;
                padding: 8px 12px;
                border-radius: 5px;
                background-color: #f9f9f9;
                transition: background-color 0.3s ease;
              }}
              .back-button:hover {{
                background-color: #e0e0e0;
              }}
              .wide-container {{
                max-width: 85% !important;
              }}
            </style>
            <div class="prediction-table-wrapper">
              <div class="table-responsive">
                {html_table}
              </div>
            </div>
            """
            print('Number lines',num_lines)
            print('Number rows', len(body_rows))

            return render_template_string(base_css + """
            <a href="{{ url_for('index') }}" class="back-button" title="Back to Home">
              &#x1F3E0;
            </a>
            <div class="container my-4 wide-container">
                <div class="mt-4">
                    <a href="{{ url_for('predict_best_threshold') }}" class="btn btn-primary">Try Again</a>
                </div>
                <h2 class="mb-3">Prediction with Best Profit Threshold</h2>
                <div class="alert alert-success">
                    <strong>Best threshold from profit curve:</strong> {{ threshold }}
                </div>
                {{ styled_table|safe }}
            </div>
            """, threshold=threshold, styled_table=styled_table)

        except Exception as e:
            print('Error on prediction', e)
            flash(f"Prediction failed: {e}", "error")
            return redirect(url_for("predict_best_threshold"))

    return render_template_string(base_css + """
    <style>
      .back-button {
          position: fixed;
          top: 10px;
          left: 10px;
          text-decoration: none;
          font-size: 16px;
          color: #333;
          border: 1px solid #ccc;
          padding: 8px 12px;
          border-radius: 5px;
          background-color: #f9f9f9;
          transition: background-color 0.3s ease;
          z-index: 1000;
      }
      .back-button:hover {
          background-color: #e0e0e0;
      }
      .wide-container {
          max-width: 85% !important;
      }
    </style>

    <a href="{{ url_for('index') }}" class="back-button" title="Back to Home">
      &#x1F3E0;
    </a>
    <div class="container my-4 wide-container">
      <h2>Predict Churn Using Best Profit Threshold</h2>
      <form method="POST" class="row g-3">
        <div class="col-md-6 form-check mt-4">
          <input type="checkbox" class="form-check-input" id="abc_assignment" name="abc_assignment" />
          <label class="form-check-label" for="abc_assignment">Generate ABC Test Groups</label>
          <br><br>
        </div>
        <div class="col-md-6">
          <label for="num_lines" class="form-label">Number of Rows</label>
          <input type="number" min="1" class="form-control" id="num_lines" name="num_lines" placeholder="100">
        </div>
        <div class="col-md-6">
          <label for="City" class="form-label">City</label>
          <input type="text" class="form-control" id="City" name="City" />
        </div>

        <div class="col-md-6">
          <label for="Gender" class="form-label">Gender</label>
          <select class="form-select" id="Gender" name="Gender">
            <option value="" selected>Choose...</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Senior_Citizen" class="form-label">Senior Citizen</label>
          <select class="form-select" id="Senior_Citizen" name="Senior_Citizen">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Partner" class="form-label">Partner</label>
          <select class="form-select" id="Partner" name="Partner">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Dependents" class="form-label">Dependents</label>
          <select class="form-select" id="Dependents" name="Dependents">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Tenure_Months" class="form-label">Tenure Months</label>
          <input type="number" min="0" class="form-control" id="Tenure_Months" name="Tenure_Months" />
        </div>

        <div class="col-md-6">
          <label for="Phone_Service" class="form-label">Phone Service</label>
          <select class="form-select" id="Phone_Service" name="Phone_Service">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Multiple_Lines" class="form-label">Multiple Lines</label>
          <select class="form-select" id="Multiple_Lines" name="Multiple_Lines">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No phone service">No phone service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Internet_Service" class="form-label">Internet Service</label>
          <select class="form-select" id="Internet_Service" name="Internet_Service">
            <option value="" selected>Choose...</option>
            <option value="DSL">DSL</option>
            <option value="Fiber optic">Fiber optic</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Online_Security" class="form-label">Online Security</label>
          <select class="form-select" id="Online_Security" name="Online_Security">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Online_Backup" class="form-label">Online Backup</label>
          <select class="form-select" id="Online_Backup" name="Online_Backup">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Device_Protection" class="form-label">Device Protection</label>
          <select class="form-select" id="Device_Protection" name="Device_Protection">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Tech_Support" class="form-label">Tech Support</label>
          <select class="form-select" id="Tech_Support" name="Tech_Support">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Streaming_TV" class="form-label">Streaming TV</label>
          <select class="form-select" id="Streaming_TV" name="Streaming_TV">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Streaming_Movies" class="form-label">Streaming Movies</label>
          <select class="form-select" id="Streaming_Movies" name="Streaming_Movies">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Contract" class="form-label">Contract</label>
          <select class="form-select" id="Contract" name="Contract">
            <option value="" selected>Choose...</option>
            <option value="Month-to-month">Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Paperless_Billing" class="form-label">Paperless Billing</label>
          <select class="form-select" id="Paperless_Billing" name="Paperless_Billing">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Payment_Method" class="form-label">Payment Method</label>
          <select class="form-select" id="Payment_Method" name="Payment_Method">
            <option value="" selected>Choose...</option>
            <option value="Electronic check">Electronic check</option>
            <option value="Mailed check">Mailed check</option>
            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
            <option value="Credit card (automatic)">Credit card (automatic)</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Monthly_Charges" class="form-label">Monthly Charges</label>
          <input
            type="number"
            step="0.01"
            min="0"
            class="form-control"
            id="Monthly_Charges"
            name="Monthly_Charges"
          />
        </div>

        <div class="col-md-6">
          <label for="Total_Charges" class="form-label">Total Charges</label>
          <input
            type="number"
            step="0.01"
            min="0"
            class="form-control"
            id="Total_Charges"
            name="Total_Charges"
          />
        </div>

        <div class="col-12 mt-3">
          <button type="submit" class="btn btn-success">Predict</button>
        </div>
      </form>
    </div>
    """)



@application.route("/predict_xai", methods=["GET", "POST"])
@login_required
def predict_xai():
    if request.method == "POST":
        features = extract_features_from_form(request.form)
        print('Extracted features for XAI:', features)

        try:
            num_lines = int(request.form.get('num_lines', 100))
            if num_lines <= 0:
                num_lines = 100

        except ValueError:
            num_lines = 100

        try:
            global_xai = request.form.get("global_xai") == "on"
            local_xai = request.form.get("local_xai") == "on"
            index_local = int(request.form.get("index_local") or 0)
            threshold_input = float(request.form.get("threshold_input") or 0.5)

            html_table, global_xai_img, local_xai_img = predict_with_xai_app(
                **features,
                threshold_input=threshold_input,
                global_xai=global_xai,
                local_xai=local_xai,
                index_local=index_local
            )

            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')

            if table:
                rows = table.find_all('tr')
                header = rows[0]
                body_rows = rows[1:num_lines + 1]
                new_table = [header] + body_rows
                table.clear()

                for row in new_table:
                    table.append(row)

                html_table = str(table)

            styled_table = f"""
            <style>
              .prediction-table-wrapper {{
                background: #ffffff;
                border-radius: 8px;
                padding: 1.5rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow-x: auto;
                max-width: 100%;
              }}
              table.table {{
                width: 100% !important;
                border-collapse: collapse;
                font-size: 0.9rem;
              }}
              table.table th, table.table td {{
                white-space: nowrap;
                text-align: center;
                padding: 0.5rem 0.75rem;
              }}
              .back-button {{
                position: fixed;
                top: 10px;
                left: 10px;
                text-decoration: none;
                font-size: 16px;
                color: #333;
                border: 1px solid #ccc;
                padding: 8px 12px;
                border-radius: 5px;
                background-color: #f9f9f9;
                transition: background-color 0.3s ease;
              }}
              .back-button:hover {{
                background-color: #e0e0e0;
              }}
              .wide-container {{
                max-width: 85% !important;
              }}
            </style>
            <div class="prediction-table-wrapper">
              <div class="table-responsive">
                {html_table}
              </div>
            </div>
            """

            return render_template_string(base_css + """
            <a href="{{ url_for('index') }}" class="back-button" title="Back to Home">
              &#x1F3E0;
            </a>
            <div class="container my-4 wide-container">
                <div class="mt-4">
                    <a href="{{ url_for('predict_xai') }}" class="btn btn-primary">Try Again</a>
                </div>
                <h2 class="mb-3">Predict Churn with XAI Options</h2>
                <div class="alert alert-success">
                    <strong>Prediction successful!</strong><br>
                    {% if global_xai %}Global XAI generated ✔️<br>{% endif %}
                    {% if local_xai %}Local XAI (index {{ index_local }}) generated ✔️<br>{% endif %}
                </div>
                {{ styled_table|safe }}
                {% if global_xai or local_xai %}
                <div class="xai-plots mt-4">
                    {% if global_xai %}
                    <h3>Global Feature Importance</h3>
                    <img src="{{ global_xai_img }}" class="img-fluid mb-4"/>
                    {% endif %}
                    {% if local_xai %}
                    <h3>Local Explanation (index {{ index_local }})</h3>
                    <img src="{{ local_xai_img }}" class="img-fluid mb-4"/>
                    {% endif %}
                </div>
                {% endif %}
            </div>
            """, styled_table=styled_table, global_xai=global_xai, local_xai=local_xai, index_local=index_local,
            global_xai_img=global_xai_img, local_xai_img=local_xai_img)

        except Exception as e:
            print('Error during XAI prediction:', e)
            flash(f"Prediction failed: {e}", "error")
            return redirect(url_for("predict_xai"))

    return render_template_string(base_css + """
    <style>
      .back-button {
          position: fixed;
          top: 10px;
          left: 10px;
          text-decoration: none;
          font-size: 16px;
          color: #333;
          border: 1px solid #ccc;
          padding: 8px 12px;
          border-radius: 5px;
          background-color: #f9f9f9;
          transition: background-color 0.3s ease;
          z-index: 1000;
      }
      .back-button:hover {
          background-color: #e0e0e0;
      }
      .wide-container {
          max-width: 85% !important;
      }
    </style>

    <a href="{{ url_for('index') }}" class="back-button" title="Back to Home">
      &#x1F3E0;
    </a>

    <div class="container my-4 wide-container">
      <h2>Predict Churn with Explainable AI (XAI)</h2>
      <form method="POST" class="row g-3">
        <div class="col-md-6">
          <label for="num_lines" class="form-label">Number of Rows</label>
          <input type="number" min="1" class="form-control" id="num_lines" name="num_lines" placeholder="100">
        </div>
        <div class="col-md-6">
          <label for="City" class="form-label">City</label>
          <input type="text" class="form-control" id="City" name="City" />
        </div>

        <div class="col-md-6">
          <label for="Gender" class="form-label">Gender</label>
          <select class="form-select" id="Gender" name="Gender">
            <option value="" selected>Choose...</option>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Senior_Citizen" class="form-label">Senior Citizen</label>
          <select class="form-select" id="Senior_Citizen" name="Senior_Citizen">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Partner" class="form-label">Partner</label>
          <select class="form-select" id="Partner" name="Partner">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Dependents" class="form-label">Dependents</label>
          <select class="form-select" id="Dependents" name="Dependents">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Tenure_Months" class="form-label">Tenure Months</label>
          <input type="number" min="0" class="form-control" id="Tenure_Months" name="Tenure_Months" />
        </div>

        <div class="col-md-6">
          <label for="Phone_Service" class="form-label">Phone Service</label>
          <select class="form-select" id="Phone_Service" name="Phone_Service">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Multiple_Lines" class="form-label">Multiple Lines</label>
          <select class="form-select" id="Multiple_Lines" name="Multiple_Lines">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No phone service">No phone service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Internet_Service" class="form-label">Internet Service</label>
          <select class="form-select" id="Internet_Service" name="Internet_Service">
            <option value="" selected>Choose...</option>
            <option value="DSL">DSL</option>
            <option value="Fiber optic">Fiber optic</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Online_Security" class="form-label">Online Security</label>
          <select class="form-select" id="Online_Security" name="Online_Security">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Online_Backup" class="form-label">Online Backup</label>
          <select class="form-select" id="Online_Backup" name="Online_Backup">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Device_Protection" class="form-label">Device Protection</label>
          <select class="form-select" id="Device_Protection" name="Device_Protection">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Tech_Support" class="form-label">Tech Support</label>
          <select class="form-select" id="Tech_Support" name="Tech_Support">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Streaming_TV" class="form-label">Streaming TV</label>
          <select class="form-select" id="Streaming_TV" name="Streaming_TV">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Streaming_Movies" class="form-label">Streaming Movies</label>
          <select class="form-select" id="Streaming_Movies" name="Streaming_Movies">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Contract" class="form-label">Contract</label>
          <select class="form-select" id="Contract" name="Contract">
            <option value="" selected>Choose...</option>
            <option value="Month-to-month">Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Paperless_Billing" class="form-label">Paperless Billing</label>
          <select class="form-select" id="Paperless_Billing" name="Paperless_Billing">
            <option value="" selected>Choose...</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Payment_Method" class="form-label">Payment Method</label>
          <select class="form-select" id="Payment_Method" name="Payment_Method">
            <option value="" selected>Choose...</option>
            <option value="Electronic check">Electronic check</option>
            <option value="Mailed check">Mailed check</option>
            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
            <option value="Credit card (automatic)">Credit card (automatic)</option>
          </select>
        </div>

        <div class="col-md-6">
          <label for="Monthly_Charges" class="form-label">Monthly Charges</label>
          <input
            type="number"
            step="0.01"
            min="0"
            class="form-control"
            id="Monthly_Charges"
            name="Monthly_Charges"
          />
        </div>

        <div class="col-md-6">
          <label for="Total_Charges" class="form-label">Total Charges</label>
          <input
            type="number"
            step="0.01"
            min="0"
            class="form-control"
            id="Total_Charges"
            name="Total_Charges"
          />
        </div>

        <div class="col-md-6">
          <label for="Threshold" class="form-label">Prediction Threshold</label>
          <input type="number" step="0.01" min="0" max="1" class="form-control" id="Threshold" name="threshold_input" value="0.5" />
        </div>

        <div class="col-md-6 form-check mt-4">
          <input type="checkbox" class="form-check-input" id="global_xai" name="global_xai" />
          <label class="form-check-label" for="global_xai">Generate Global XAI</label>
        </div>

        <div class="col-md-6 form-check mt-4">
          <input type="checkbox" class="form-check-input" id="local_xai" name="local_xai" />
          <label class="form-check-label" for="local_xai">Generate Local XAI</label>
        </div>

        <div class="col-md-6">
          <label for="index_local" class="form-label">Local XAI Index</label>
          <input type="number" min="0" class="form-control" id="index_local" name="index_local" value="0" />
        </div>

        <div class="col-12 mt-3">
          <button type="submit" class="btn btn-success">Predict with XAI</button>
        </div>
      </form>
    </div>
    """)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    application.run(host='0.0.0.0', port=port, debug=False)