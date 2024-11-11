from flask import Flask, redirect, render_template, request, url_for, session
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import scipy.stats as stats

app = Flask(__name__)
app.secret_key = "not_very_secret_key"


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # 3: Fit a linear regression model to X and Y
    model = LinearRegression()  # Initialize the LinearRegression model
    model.fit(X.reshape(-1, 1), Y)  # Fit the model to X and Y
    slope = model.coef_[0]  # Extract the slope (coefficient) from the fitted model
    intercept = model.intercept_  # Extract the intercept from the fitted model

    # 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label=f'Y = {slope:.4f}X + {intercept:.4f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot1_path)
    plt.close()

    # 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)

        # 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        intercepts.append(sim_intercept)

    # 8: Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(slopes, bins=30, color='#89CFF0', edgecolor=None, alpha=0.7)
    plt.axvline(slope, color='red', linestyle='dashed', linewidth=2, label='Observed Slope')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.title('Histogram of Slopes')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(intercepts, bins=30, color='#FFB347', edgecolor=None, alpha=0.7)
    plt.axvline(intercept, color='red', linestyle='dashed', linewidth=2, label='Observed Intercept')
    plt.xlabel('Intercept')
    plt.ylabel('Frequency')
    plt.title('Histogram of Intercepts')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    # 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= np.abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= np.abs(intercept))

    # Return data needed for further analysis
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

            if N <= 0 or S <= 0 or sigma2 < 0:
                raise ValueError

        except (ValueError, KeyError):
            return render_template("index.html", error="Invalid input. Please enter valid numerical values.")

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
    return render_template("index.html")


@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "GET":
        return redirect(url_for("index"))
    return index()

@app.route("/hypothesis_test", methods=["GET", "POST"])
def hypothesis_test():
    if request.method == "GET":
        return redirect(url_for("index"))
    # Retrieve data from session
    try:
        N = int(session.get("N"))
        S = int(session.get("S"))
        slope = float(session.get("slope"))
        intercept = float(session.get("intercept"))
        slopes = session.get("slopes")
        intercepts = session.get("intercepts")
        beta0 = float(session.get("beta0"))
        beta1 = float(session.get("beta1"))
    except (TypeError, ValueError):
        return render_template("index.html", error="Session data is missing or corrupted.")

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if parameter not in ["slope", "intercept"] or test_type not in [">", "<", "!="]:
        return render_template("index.html", error="Invalid test parameters.")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
        param_label = r"$\beta_1$"
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0
        param_label = r"$\beta_0$"

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == "!=":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
    else:
        p_value = None

    # If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    if p_value is not None and p_value <= 0.0001:
        fun_message = "Wow! That's an extremely rare event!"
    else:
        fun_message = None

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=30, color='#89CFF0', edgecolor=None, alpha=0.7, label="Simulated Statistics")
    
    # Mark the observed statistic
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2,
                label=f'Observed {parameter.capitalize()} ({observed_stat:.4f})')

    # Mark the hypothesized parameter value
    plt.axvline(hypothesized_value, color='purple', linestyle=':', linewidth=2,
                label=f'Hypothesized {parameter.capitalize()} ({param_label} = {hypothesized_value:.4f})')

    # Add labels, title, and legend
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Simulated {parameter.capitalize()}s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot3_path)
    plt.close()

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


@app.route("/confidence_interval", methods=["GET", "POST"])
def confidence_interval():
    if request.method == "GET":
        return redirect(url_for("index"))
    try:
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
        slopes = session.get("slopes")
        intercepts = session.get("intercepts")
    except (TypeError, ValueError):
        return render_template("index.html", error="Session data is missing or corrupted.")

    parameter = request.form.get("parameter")
    confidence_level = request.form.get("confidence_level")

    try:
        confidence_level = float(confidence_level)
        if confidence_level not in [90, 95, 99]:
            raise ValueError
    except (ValueError, TypeError):
        return render_template("index.html", error="Invalid confidence level selected.")

    if parameter not in ["slope", "intercept"]:
        return render_template("index.html", error="Invalid parameter selected.")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
        param_symbol = r'$\beta_1$'
        mean_symbol = r'$\overline{\beta_1}$'
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0
        param_symbol = r'$\beta_0$'
        mean_symbol = r'$\overline{\beta_0}$'

    # 14: Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # 15: Calculate confidence interval for the parameter estimate
    # Use the t-distribution and confidence_level
    alpha = 1 - (confidence_level / 100)
    t_critical = stats.t.ppf(1 - alpha / 2, df=S - 1)
    ci_lower = mean_estimate - t_critical * (std_estimate / np.sqrt(S))
    ci_upper = mean_estimate + t_critical * (std_estimate / np.sqrt(S))

    # 16: Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # 17: Plot the individual estimates as gray points and confidence interval
    # Plot the mean estimate as a colored point which changes if the true parameter is included
    # Plot the confidence interval as a horizontal line
    # Plot the true parameter value on the plot
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 2))

    # Plot individual simulated estimates as gray points
    plt.scatter(estimates, np.zeros_like(estimates), color='gray', alpha=0.5, label='Simulated Estimates')

    # Plot the mean estimate as a colored point
    plt.scatter(mean_estimate, 0, color='blue', s=100, label=f'Mean Estimate ({mean_symbol})')

    # Plot the confidence interval as a horizontal line
    ci_color = 'green' if includes_true else 'red'
    plt.hlines(0, ci_lower, ci_upper, colors=ci_color, linewidth=4, label=f'{confidence_level}% Confidence Interval')

    # Plot the true parameter value as a vertical line
    plt.axvline(true_param, color='purple', linestyle=':', linewidth=2, label=f'True {parameter.capitalize()} ({param_symbol})')

    # Remove y-axis ticks and labels
    plt.yticks([])
    plt.ylabel('')

    # Set x-axis label and title
    plt.xlabel(f'{parameter.capitalize()} Value')
    plt.title(f'{confidence_level}% Confidence Interval for {parameter.capitalize()}')

    # Add legend with equations
    plt.legend(loc='upper right', fontsize=10)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(plot4_path)
    plt.close()

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


if __name__ == "__main__":
    app.run(debug=True)
