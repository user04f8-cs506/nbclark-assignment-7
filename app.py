from flask import Flask, render_template, request, url_for, session, flash, redirect
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import os
import logging

app = Flask(__name__)

# **Security Enhancement: Use environment variable for secret key**
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default_secret_key")  # Replace with a secure key in production

# **Logging Configuration**
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

def validate_generate_data_inputs(N, mu, sigma2, beta0, beta1, S):
    """
    Validates inputs for data generation.
    Returns a tuple (is_valid, message).
    """
    if N <= 0:
        return False, "Sample size (N) must be a positive integer."
    if sigma2 <= 0:
        return False, "Variance (σ²) must be a positive number."
    if S <= 0:
        return False, "Number of simulations (S) must be a positive integer."
    return True, ""

def validate_hypothesis_test_inputs(parameter, test_type):
    """
    Validates inputs for hypothesis testing.
    Returns a tuple (is_valid, message).
    """
    valid_parameters = {"slope", "intercept"}
    valid_test_types = {"two-tailed", "greater", "less"}

    if parameter not in valid_parameters:
        return False, "Invalid parameter selected for hypothesis testing."
    if test_type not in valid_test_types:
        return False, "Invalid test type selected."
    return True, ""

def validate_confidence_interval_inputs(parameter, confidence_level):
    """
    Validates inputs for confidence interval calculation.
    Returns a tuple (is_valid, message).
    """
    valid_parameters = {"slope", "intercept"}
    valid_confidence_levels = {90, 95, 99}

    if parameter not in valid_parameters:
        return False, "Invalid parameter selected for confidence interval."
    if confidence_level not in valid_confidence_levels:
        return False, "Invalid confidence level selected."
    return True, ""

def generate_scatter_plot(X, Y, model, plot_path):
    """
    Generates and saves a scatter plot with the regression line.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def generate_histograms(slopes, intercepts, observed_slope, observed_intercept, plot_path):
    """
    Generates and saves histograms for slopes and intercepts.
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(slopes, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(observed_slope, color='red', linestyle='dashed', linewidth=2, label='Observed Slope')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.title('Histogram of Slopes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(intercepts, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(observed_intercept, color='red', linestyle='dashed', linewidth=2, label='Observed Intercept')
    plt.xlabel('Intercept')
    plt.ylabel('Frequency')
    plt.title('Histogram of Intercepts')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def generate_histogram_simulated_stats(simulated_stats, observed_stat, parameter, plot_path):
    """
    Generates and saves a histogram of simulated statistics with the observed statistic.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Simulated {parameter.capitalize()}s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def generate_confidence_interval_plot(estimates, mean_estimate, ci_lower, ci_upper, true_param, includes_true, parameter, plot_path):
    """
    Generates and saves a plot for confidence interval results.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(1, len(estimates)+1), estimates, color='gray', alpha=0.5, label='Estimates')
    plt.axhline(mean_estimate, color='blue', linestyle='-', label='Mean Estimate')
    plt.axhline(ci_lower, color='green', linestyle='--', label='Confidence Interval')
    plt.axhline(ci_upper, color='green', linestyle='--')
    if includes_true:
        plt.scatter(len(estimates)+1, true_param, color='orange', label='True Parameter')
    else:
        plt.scatter(len(estimates)+1, true_param, color='red', label='True Parameter')
    plt.xlabel('Simulation')
    plt.ylabel(parameter.capitalize())
    plt.title(f'Confidence Interval for {parameter.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def generate_data(N, mu, beta0, beta1, sigma2, S):
    """
    Generates the dataset, fits the model, runs simulations, and creates plots.
    """
    try:
        logging.info("Generating data with N=%d, mu=%.2f, beta0=%.2f, beta1=%.2f, sigma2=%.2f, S=%d",
                     N, mu, beta0, beta1, sigma2, S)

        # TODONE 1: Generate a random dataset X of size N with values between 0 and 1
        X = np.random.rand(N)

        # TODONE 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
        Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), size=N)

        # TODONE 3: Fit a linear regression model to X and Y
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), Y)
        slope = model.coef_[0]
        intercept = model.intercept_

        # TODONE 4: Generate a scatter plot of (X, Y) with the fitted regression line
        plot1_path = "static/plot1.png"
        generate_scatter_plot(X, Y, model, plot1_path)

        # TODONE 5: Run S simulations to generate slopes and intercepts
        slopes = []
        intercepts = []

        for sim in range(S):
            # TODONE 6: Generate simulated datasets using the same beta0 and beta1
            X_sim = np.random.rand(N)
            Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), size=N)

            # TODONE 7: Fit linear regression to simulated data and store slope and intercept
            sim_model = LinearRegression()
            sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
            sim_slope = sim_model.coef_[0]
            sim_intercept = sim_model.intercept_

            slopes.append(sim_slope)
            intercepts.append(sim_intercept)

            if (sim + 1) % 1000 == 0:
                logging.info("Completed %d/%d simulations", sim + 1, S)

        # TODONE 8: Plot histograms of slopes and intercepts
        plot2_path = "static/plot2.png"
        generate_histograms(slopes, intercepts, slope, intercept, plot2_path)

        # TODONE 9: Calculate proportions of slopes and intercepts more extreme than observed
        slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
        intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

        logging.info("Proportion of slopes more extreme: %.4f", slope_more_extreme)
        logging.info("Proportion of intercepts more extreme: %.4f", intercept_extreme)

        return (
            X,
            Y,
            slope,
            intercept,
            plot1_path,
            plot2_path,
            slope_more_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        )
    except Exception as e:
        logging.error("Error in generate_data: %s", str(e))
        raise

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        try:
            N = int(request.form["N"])
            mu = float(request.form["mu"])
            sigma2 = float(request.form["sigma2"])
            beta0 = float(request.form["beta0"])
            beta1 = float(request.form["beta1"])
            S = int(request.form["S"])

            # Input Validation
            is_valid, message = validate_generate_data_inputs(N, mu, sigma2, beta0, beta1, S)
            if not is_valid:
                flash(message, "error")
                return redirect(url_for('index'))

            # Generate data and initial plots
            (
                X,
                Y,
                slope,
                intercept,
                plot1,
                plot2,
                slope_extreme,
                intercept_extreme,
                slopes,
                intercepts,
            ) = generate_data(N, mu, beta0, beta1, sigma2, S)

            # Store data in session
            session["X"] = X.tolist()
            session["Y"] = Y.tolist()
            session["slope"] = slope
            session["intercept"] = intercept
            session["slopes"] = slopes
            session["intercepts"] = intercepts
            session["slope_extreme"] = slope_extreme
            session["intercept_extreme"] = intercept_extreme
            session["N"] = N
            session["mu"] = mu
            session["sigma2"] = sigma2
            session["beta0"] = beta0
            session["beta1"] = beta1
            session["S"] = S

            logging.info("Data generation successful.")

            # Return render_template with variables
            return render_template(
                "index.html",
                plot1=plot1,
                plot2=plot2,
                slope_extreme=slope_extreme,
                intercept_extreme=intercept_extreme,
                N=N,
                mu=mu,
                sigma2=sigma2,
                beta0=beta0,
                beta1=beta1,
                S=S,
            )
        except ValueError as ve:
            logging.error("ValueError: %s", str(ve))
            flash("Invalid input. Please ensure all fields are filled correctly.", "error")
            return redirect(url_for('index'))
        except Exception as e:
            logging.error("Exception in index route: %s", str(e))
            flash("An unexpected error occurred during data generation.", "error")
            return redirect(url_for('index'))

    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_route():
    # This route handles data generation (same as above)
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    try:
        # Retrieve data from session
        N = int(session.get("N"))
        S = int(session.get("S"))
        slope = float(session.get("slope"))
        intercept = float(session.get("intercept"))
        slopes = session.get("slopes")
        intercepts = session.get("intercepts")
        beta0 = float(session.get("beta0"))
        beta1 = float(session.get("beta1"))

        # Convert lists back to numpy arrays
        slopes = np.array(slopes)
        intercepts = np.array(intercepts)

        parameter = request.form.get("parameter")
        test_type = request.form.get("test_type")

        # Input Validation
        is_valid, message = validate_hypothesis_test_inputs(parameter, test_type)
        if not is_valid:
            flash(message, "error")
            return redirect(url_for('index'))

        # Use the slopes or intercepts from the simulations
        if parameter == "slope":
            simulated_stats = slopes
            observed_stat = slope
            hypothesized_value = beta1
        else:
            simulated_stats = intercepts
            observed_stat = intercept
            hypothesized_value = beta0

        # TODONE 10: Calculate p-value based on test type
        if test_type == "two-tailed":
            p_value = np.mean(np.abs(simulated_stats) >= np.abs(observed_stat))
        elif test_type == "greater":
            p_value = np.mean(simulated_stats >= observed_stat)
        elif test_type == "less":
            p_value = np.mean(simulated_stats <= observed_stat)
        else:
            p_value = None  # Undefined test type

        # TODONE 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
        if p_value is not None and p_value <= 0.0001:
            fun_message = "Wow! That's extremely significant! 🎉"
        else:
            fun_message = None

        # TODONE 12: Plot histogram of simulated statistics
        plot3_path = "static/plot3.png"
        generate_histogram_simulated_stats(simulated_stats, observed_stat, parameter, plot3_path)

        logging.info("Hypothesis testing completed: p_value=%.4f", p_value)

        # Return results to template
        return render_template(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot3=plot3_path,
            parameter=parameter,
            observed_stat=observed_stat,
            hypothesized_value=hypothesized_value,
            N=N,
            beta0=beta0,
            beta1=beta1,
            S=S,
            p_value=p_value,
            fun_message=fun_message,
        )
    except KeyError as ke:
        logging.error("KeyError: %s", str(ke))
        flash("Session data missing. Please generate data first.", "error")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error("Exception in hypothesis_test route: %s", str(e))
        flash("An unexpected error occurred during hypothesis testing.", "error")
        return redirect(url_for('index'))

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        # Retrieve data from session
        N = int(session.get("N"))
        mu = float(session.get("mu"))
        sigma2 = float(session.get("sigma2"))
        beta0 = float(session.get("beta0"))
        beta1 = float(session.get("beta1"))
        S = int(session.get("S"))
        X = np.array(session.get("X"))
        Y = np.array(session.get("Y"))
        slope = float(session.get("slope"))
        intercept = float(session.get("intercept"))
        slopes = np.array(session.get("slopes"))
        intercepts = np.array(session.get("intercepts"))

        parameter = request.form.get("parameter")
        confidence_level = float(request.form.get("confidence_level"))

        # Input Validation
        is_valid, message = validate_confidence_interval_inputs(parameter, confidence_level)
        if not is_valid:
            flash(message, "error")
            return redirect(url_for('index'))

        # Use the slopes or intercepts from the simulations
        if parameter == "slope":
            estimates = slopes
            observed_stat = slope
            true_param = beta1
        else:
            estimates = intercepts
            observed_stat = intercept
            true_param = beta0

        # TODONE 14: Calculate mean and standard deviation of the estimates
        mean_estimate = np.mean(estimates)
        std_estimate = np.std(estimates, ddof=1)

        # TODONE 15: Calculate confidence interval for the parameter estimate
        confidence_interval = stats.t.interval(
            confidence_level / 100,  # Convert percentage to proportion
            df=S-1,
            loc=mean_estimate,
            scale=std_estimate / np.sqrt(S)
        )
        ci_lower, ci_upper = confidence_interval

        # TODONE 16: Check if confidence interval includes true parameter
        includes_true = ci_lower <= true_param <= ci_upper

        # TODONE 17: Plot the individual estimates as gray points and confidence interval
        plot4_path = "static/plot4.png"
        generate_confidence_interval_plot(estimates, mean_estimate, ci_lower, ci_upper,
                                          true_param, includes_true, parameter, plot4_path)

        logging.info("Confidence interval calculated: %.2f%% CI [%.4f, %.4f], includes_true=%s",
                     confidence_level, ci_lower, ci_upper, includes_true)

        # Return results to template
        return render_template(
            "index.html",
            plot1="static/plot1.png",
            plot2="static/plot2.png",
            plot4=plot4_path,
            parameter=parameter,
            confidence_level=confidence_level,
            mean_estimate=mean_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            includes_true=includes_true,
            observed_stat=observed_stat,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    except KeyError as ke:
        logging.error("KeyError: %s", str(ke))
        flash("Session data missing. Please generate data first.", "error")
        return redirect(url_for('index'))
    except Exception as e:
        logging.error("Exception in confidence_interval route: %s", str(e))
        flash("An unexpected error occurred during confidence interval calculation.", "error")
        return redirect(url_for('index'))

if __name__ == "__main__":
    # **Security Enhancement: Do not use debug=True in production**
    app.run(debug=True)
