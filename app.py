from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years",
        "Average number of years a newborn would live if current age-specific mortality rates remained constant."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force",
        "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person",
        "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year",
        "Annual number of live births per 1,000 people. This is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people",
        "Net number of migrants during the period (immigrants minus emigrants); it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "deaths per 100,000 people",
        "Intentional homicide victims per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual",
        "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP",
        "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US$",
        "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP",
        "External balance on goods and services as a percentage of GDP."),
}
WB_API = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
WB_PAGE = "https://data.worldbank.org/indicator/{indicator}?locations={country}"
CONTEXTS = {
    "Business": ("Revenue", "dollars", "time", "years", "a business revenue model", "a*exp(k*x)"),
    "Engineering": ("Position", "meters", "time", "seconds", "an object's position model", "a*x**2+b*x+c"),
    "Medical": ("Concentration", "mg/L", "time", "hours", "a concentration-over-time model", "a*exp(-k*x)"),
    "Sports science": ("Speed", "m/s", "time", "seconds", "an athlete's speed model", "a*sin(b*x)+c"),
    "Biology": ("Population", "organisms", "time", "days", "a population growth model", "a*exp(k*x)"),
    "Bioengineering": ("Signal", "volts", "time", "seconds", "a measured biological signal", "a*sin(b*x)+c"),
    "Economic forecasting": ("Index", "index points", "time", "years", "an economic index model", "a*x**2+b*x+c"),
    "Weather": ("Temperature", "°C", "time", "hours", "a temperature model", "a*sin(b*x)+c"),
    "Custom": ("Output", "units", "input", "units", "a custom model", "a*x**2+b*x+c"),
}
FUNCTION_TYPES = ["Polynomial", "Rational", "Exponential", "Logarithmic", "Trigonometric"]
SIMULATIONS = ["Walking / prosthetic-leg gait", "Moving object / projectile", "Tank / concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"]

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("Hispanic Heritage Data Lab • authentic historical data • mathematical modeling • calculus")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country: str, indicator: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str]:
    url = WB_API.format(country=country, indicator=indicator)
    params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=25, headers={"User-Agent": "HispanicHeritageDataExplorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            return pd.DataFrame(columns=["Year", "Value"]), "The World Bank returned no observations for this selection."
        rows = [{"Year": int(item["date"]), "Value": item.get("value")}
                for item in payload[1] if item.get("date") and item.get("value") is not None]
        frame = pd.DataFrame(rows, columns=["Year", "Value"])
        if frame.empty:
            return frame, "The API response contained no non-null annual values."
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        frame = frame.dropna().sort_values("Year").drop_duplicates("Year").reset_index(drop=True)
        return frame, ""
    except requests.Timeout:
        return pd.DataFrame(columns=["Year", "Value"]), "The World Bank request timed out. Try again later."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["Year", "Value"]), f"World Bank network/API error: {exc}"
    except (ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["Year", "Value"]), f"Could not interpret the World Bank response: {exc}"

def indicator_validity(name: str, value: float) -> bool:
    if not np.isfinite(value):
        return False
    if name == "Life expectancy at birth":
        return 0 <= value <= 125
    if name == "Unemployment, national estimate":
        return 0 <= value <= 100
    if name == "GDP per capita" or name == "Total reserves including gold":
        return value >= 0
    if name == "Crude birth rate" or name == "Intentional homicide rate":
        return value >= 0
    if name == "Central government debt":
        return value >= 0
    return True

def validate_working_data(frame: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    if not {"Year", "Value"}.issubset(frame.columns):
        return pd.DataFrame(columns=["Year", "Value"]), ["The editable table must contain Year and Value columns."]
    out = frame[["Year", "Value"]].copy()
    numeric_year = pd.to_numeric(out["Year"], errors="coerce")
    numeric_val = pd.to_numeric(out["Value"], errors="coerce")
    if numeric_year.isna().any() or ((numeric_year % 1) != 0).any():
        errors.append("Every year must be a whole-number integer.")
    if numeric_val.isna().any():
        errors.append("Every value must be numeric; remove blank or nonnumeric rows.")
    out["Year"], out["Value"] = numeric_year, numeric_val
    out = out.dropna()
    if out["Year"].duplicated().any():
        errors.append("Duplicate years are not allowed. Keep one observation per year.")
    if not out.empty and ((out["Year"] < 1800) | (out["Year"] > date.today().year)).any():
        errors.append("Years must be between 1800 and the current calendar year.")
    if not out.empty:
        bad = [float(v) for v in out["Value"] if not indicator_validity(indicator_name, float(v))]
        if bad:
            errors.append(f"{indicator_name} has indicator-specific invalid values (examples: {bad[:3]}).")
    out = out.sort_values("Year").reset_index(drop=True)
    return out, errors

def fit_centered_polynomial(frame: pd.DataFrame, degree: int):
    clean = frame.sort_values("Year")
    years = clean["Year"].to_numpy(dtype=float)
    values = clean["Value"].to_numpy(dtype=float)
    if len(np.unique(years)) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; only {len(np.unique(years))} are available.")
    base_year = int(years.min())
    t = years - base_year
    coeffs = np.polyfit(t, values, degree)
    poly = np.poly1d(coeffs)
    predicted = poly(t)
    residuals = values - predicted
    r2 = float(r2_score(values, predicted)) if len(values) > 1 else float("nan")
    mae = float(mean_absolute_error(values, predicted))
    return {"coeffs": coeffs, "poly": poly, "base": base_year, "years": years,
            "values": values, "predicted": predicted, "residuals": residuals,
            "r2": r2, "mae": mae, "mean": float(np.mean(values)), "degree": degree}

def poly_equation(model, unit: str) -> str:
    terms = []
    for power, coefficient in enumerate(model["coeffs"][::-1]):
        if abs(coefficient) < 1e-12:
            continue
        terms.append(f"({coefficient:.10g})" if power == 0 else f"({coefficient:.10g})·t" if power == 1 else f"({coefficient:.10g})·t^{power}")
    return "y(t) = " + (" + ".join(terms[::-1]) if terms else "0") + f", where t = year − {model['base']}"

def model_predict(model, years):
    return model["poly"](np.asarray(years, dtype=float) - model["base"])

def holdout_score(frame: pd.DataFrame, degree: int):
    ordered = frame.sort_values("Year")
    n = len(ordered)
    if n < max(degree + 3, 10):
        return None
    split = max(degree + 1, int(math.floor(n * 0.8)))
    if split >= n:
        return None
    train, test = ordered.iloc[:split], ordered.iloc[split:]
    if train["Year"].nunique() < degree + 1:
        return None
    b = int(train["Year"].min())
    p = np.poly1d(np.polyfit(train["Year"].to_numpy() - b, train["Value"].to_numpy(), degree))
    pred = p(test["Year"].to_numpy() - b)
    return {"mae": float(mean_absolute_error(test["Value"], pred)),
            "r2": float(r2_score(test["Value"], pred)) if len(test) > 1 else None,
            "train_n": len(train), "test_n": len(test)}

def safe_expression(source: str):
    x = sp.symbols("x", real=True)
    allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi,
               "abs": sp.Abs}
    cleaned = source.replace("^", "**")
    if len(cleaned) > 180:
        raise ValueError("Expression is too long (maximum 180 characters).")
    try:
        expr = sp.sympify(cleaned, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not parse the expression. Use x, +, -, *, /, **, sin(), cos(), tan(), exp(), log(), sqrt(), and pi.") from exc
    if expr.free_symbols - {x}:
        raise ValueError("Only the variable x and supported mathematical functions are allowed.")
    if expr.has(sp.Integral, sp.Derivative, sp.Lambda):
        raise ValueError("That expression type is not supported.")
    return x, sp.simplify(expr)

def numerical_roots(expr, x, low, high, samples=1600):
    f = sp.lambdify(x, expr, modules=["numpy"])
    grid = np.linspace(low, high, samples)
    roots = []
    vals = []
    for z in grid:
        try:
            v = float(f(z))
            vals.append(v if np.isfinite(v) else np.nan)
        except Exception:
            vals.append(np.nan)
    vals = np.asarray(vals)
    for i in range(len(grid)-1):
        a, b, fa, fb = grid[i], grid[i+1], vals[i], vals[i+1]
        if np.isfinite(fa) and abs(fa) < 1e-6:
            roots.append(float(a))
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            lo, hi = a, b
            for _ in range(45):
                mid = (lo + hi) / 2
                try: fm = float(f(mid))
                except Exception: break
                if not np.isfinite(fm): break
                if fa * fm <= 0: hi = mid
                else: lo, fa = mid, fm
            roots.append(float((lo+hi)/2))
    unique = []
    for r in roots:
        if low - 1e-7 <= r <= high + 1e-7 and all(abs(r-u) > 1e-3 for u in unique):
            unique.append(r)
    return sorted(unique)

def evaluate_scalar(expr, x, value):
    f = sp.lambdify(x, expr, modules=["numpy"])
    try:
        result = float(f(float(value)))
        return result if np.isfinite(result) else None
    except Exception:
        return None

def fmt_num(value):
    return f"{value:,.8g}" if value is not None and np.isfinite(value) else "undefined"

def make_chart(data, model, sample_step, horizon, single_year=None, single_value=None, batch=None,
               comparison=None, x_label="Year", y_label="Value"):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(data["Year"], data["Value"], label="Observed data", color="#1565c0", s=24, zorder=4)
    lo, hi = int(data["Year"].min()), int(data["Year"].max())
    end = min(2100, hi + horizon)
    xs = np.linspace(lo, max(end, hi), max(300, (end-lo+1)*3))
    ys = model_predict(model, xs)
    ax.plot(xs, ys, label="Smooth polynomial fit", color="#ef6c00", linewidth=2)
    sample_years = np.arange(lo, hi+1, sample_step)
    ax.scatter(sample_years, model_predict(model, sample_years), marker="x",
               label=f"Model markers every {sample_step} year(s)", color="#2e7d32", s=28)
    ax.axvline(hi, color="gray", linestyle="--", label="Historical boundary")
    if end > hi:
        ext_x = np.linspace(hi, end, 100)
        ax.plot(ext_x, model_predict(model, ext_x), linestyle=":", color="#8e24aa", label="Extrapolated segment")
    if single_year is not None and single_value is not None:
        ax.scatter([single_year], [single_value], marker="*", s=150, color="#c62828", label=f"Estimate for {single_year}", zorder=6)
    if batch is not None and not batch.empty:
        ax.scatter(batch["Year"], batch["Estimate"], marker="D", s=35, color="#00838f", label="Batch estimates", zorder=5)
    if comparison is not None:
        for label, frame in comparison:
            if not frame.empty:
                ax.plot(frame["Year"], frame["Value"], marker=".", linewidth=1, alpha=.8, label=f"{label} observed")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Historical observations and fitted model")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig

def build_excel(summary: dict, working: pd.DataFrame, results: pd.DataFrame):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        working.to_excel(writer, sheet_name="Working data", index=False)
        results.to_excel(writer, sheet_name="Calculated years", index=False)
        for sheet, frame in [("Summary", pd.DataFrame([summary])), ("Working data", working), ("Calculated years", results)]:
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 0)
            for col_idx, col in enumerate(frame.columns):
                width = min(max(14, len(str(col)) + 2), 42)
                ws.set_column(col_idx, col_idx, width)
    output.seek(0)
    return output.getvalue()

def build_pdf(summary: dict, chart_bytes: bytes, preview: pd.DataFrame, results: pd.DataFrame):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=34, leftMargin=34, topMargin=34, bottomMargin=34)
    styles = getSampleStyleSheet()
    story = [Paragraph(APP_TITLE, styles["Title"]), Paragraph("Hispanic Heritage Data Lab report", styles["Heading2"]), Spacer(1, 8)]
    for key, value in summary.items():
        story.append(Paragraph(f"<b>{key}:</b> {str(value)}", styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(Image(io.BytesIO(chart_bytes), width=510, height=265))
    for title, frame in [("Data preview", preview), ("Calculated-year results", results)]:
        story.append(Paragraph(title, styles["Heading2"]))
        if frame.empty:
            story.append(Paragraph("No rows available.", styles["BodyText"]))
        else:
            view = frame.head(18).fillna("").astype(str)
            table_data = [list(view.columns)] + view.values.tolist()
            tbl = Table(table_data, repeatRows=1)
            tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dce6f1")),
                ("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),
                ("VALIGN",(0,0),(-1,-1),"TOP")]))
            story.append(tbl)
        story.append(Spacer(1, 8))
    doc.build(story)
    output.seek(0)
    return output.getvalue()

def main():
    current_year = date.today().year
    start_year, end_year = current_year - 70, current_year - 1
    with st.sidebar:
        st.header("Historical data")
        country_label = st.selectbox("Featured country", list(COUNTRIES))
        country_code = COUNTRIES[country_label]
        indicator_name = st.selectbox("Indicator", list(INDICATORS))
        indicator_code, unit, definition = INDICATORS[indicator_name]
        st.caption(f"World Bank indicator: `{indicator_code}`")
        degree = st.slider("Polynomial degree", 3, 8, 3)
        sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
        horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
        printer_mode = st.toggle("Printer-friendly view", False)
        st.divider()
        st.subheader("Mathematics pathway")
        pathway = st.selectbox("Course level", ["Honors Algebra 2", "Precalculus", "Calculus"])
        st.caption("Brazil is included as a Latin American comparison; it is not a Spanish-speaking country.")

    st.subheader("Research setup")
    research_question = st.text_input("Research question", "How has life expectancy changed over time in the selected country?")
    st.markdown(f"**Definition:** {definition}")
    st.markdown(f"**Units:** {unit}  \n**Requested period:** {start_year}–{end_year} (70 calendar years)")
    source_url = WB_PAGE.format(indicator=indicator_code, country=country_code)
    st.markdown(f"**Source:** [World Bank indicator page]({source_url}) · [API endpoint]({WB_API.format(country=country_code, indicator=indicator_code)})")
    st.caption(f"Retrieved: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}")

    with st.spinner("Requesting official World Bank observations…"):
        source_data, api_message = fetch_world_bank(country_code, indicator_code, start_year, end_year)
    if api_message:
        st.warning(api_message)
    if source_data.empty:
        st.error("No source observations are available. The app will not invent or substitute data. Check the API connection or choose another indicator/country.")
        source_data = pd.DataFrame(columns=["Year", "Value"])
    if not source_data.empty:
        st.info(f"Actual coverage: {int(source_data.Year.min())}–{int(source_data.Year.max())} · {len(source_data)} non-null annual observations (requested 70 calendar years). Missing years are not filled.")

    key = f"working_{country_code}_{indicator_code}"
    if key not in st.session_state:
        st.session_state[key] = source_data.copy()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Reset to source data"):
            st.session_state[key] = source_data.copy()
            st.session_state.pop(f"editor_{key}", None)
            st.session_state.pop("calculated_results", None)
            st.rerun()
    with col_b:
        st.caption("Edit the Year and Value columns below. Changes are treated as a user-modified scenario.")
    edited = st.data_editor(st.session_state[key], num_rows="dynamic", use_container_width=True,
                            key=f"editor_{key}",
                            column_config={"Year": st.column_config.NumberColumn("Year", format="%d"),
                                           "Value": st.column_config.NumberColumn(f"Value ({unit})", format="%.8g")})
    st.session_state[key] = edited.copy()
    validated, validation_errors = validate_working_data(edited, indicator_name)
    is_edited = not validated.reset_index(drop=True).equals(source_data.reset_index(drop=True))
    if is_edited:
        st.warning("USER-MODIFIED SCENARIO — results and exports use your edited working data, not only the original source.")
    if validation_errors:
        for error in validation_errors: st.error(error)
        st.error("Correct the working table before fitting the model.")
        st.stop()
    if len(validated) < degree + 1 or validated["Year"].nunique() < degree + 1:
        st.error(f"Cannot fit degree {degree}: need at least {degree+1} valid distinct years; currently have {validated.Year.nunique()}.")
        return
    model = fit_centered_polynomial(validated, degree)
    equation = poly_equation(model, unit)
    st.subheader("Historical model")
    st.code(equation)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (in-sample)", fmt_num(model["r2"]))
    m2.metric("MAE (in-sample)", f"{fmt_num(model['mae'])} {unit}")
    m3.metric("Observation count", len(validated))
    m4.metric("Historical mean", f"{fmt_num(model['mean'])} {unit}")
    st.caption(f"Base year: {model['base']} · Degree: {degree} · Historical range: {int(model['years'].min())}–{int(model['years'].max())}. Fit statistics are in-sample, not proof of predictive accuracy.")
    holdout = holdout_score(validated, degree)
    if holdout:
        st.write(f"**Chronological holdout (separate from in-sample fit):** trained on {holdout['train_n']} earliest observations and evaluated on {holdout['test_n']} later observations; MAE = {holdout['mae']:.6g} {unit}; R² = {fmt_num(holdout['r2'])}.")
    else:
        st.caption("Chronological holdout score not shown: insufficient observations for this degree.")
    residual_table = pd.DataFrame({"Year": model["years"].astype(int), "Actual": model["values"],
                                   "Fitted": model["predicted"], "Residual": model["residuals"]})
    with st.expander("Inspect residuals and fitted observations"):
        st.dataframe(residual_table, use_container_width=True)
    st.info(f"Uncertainty: the in-sample MAE is {model['mae']:.5g} {unit}. Residual size is a descriptive error measure, not a validated 95% prediction interval. Polynomial extrapolation can become implausible quickly; estimates outside the observed year range are especially uncertain.")

    st.subheader("Calculate values")
    c1, c2 = st.columns(2)
    with c1:
        single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=max(1950, min(2100, int(validated.Year.max()))), step=1)
        calc_clicked = st.button("Calculate value")
    with c2:
        batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
        if batch_mode == "Year range":
            r1, r2, r3 = st.columns(3)
            with r1: batch_start = st.number_input("Start", 1950, 2100, 2000, key="batch_start")
            with r2: batch_end = st.number_input("End", 1950, 2100, 2020, key="batch_end")
            with r3: batch_step = st.number_input("Step", 1, 10, 5, key="batch_step")
            custom_text = ""
        else:
            custom_text = st.text_input("Comma-separated years", "2000, 2010, 2020")
            batch_start, batch_end, batch_step = 2000, 2020, 5

    signature = (country_code, indicator_code, degree, tuple(validated["Year"].tolist()), tuple(np.round(validated["Value"].to_numpy(), 10)))
    if st.session_state.get("model_signature") != signature:
        st.session_state["model_signature"] = signature
        st.session_state.pop("calculated_results", None)
        st.session_state.pop("single_result", None)
    if calc_clicked:
        val = float(model_predict(model, [single_year])[0])
        st.session_state["single_result"] = {"Year": int(single_year), "Estimate": val,
            "Classification": "within fitted historical range" if model["years"].min() <= single_year <= model["years"].max() else "outside fitted historical range"}
    single_result = st.session_state.get("single_result")
    if single_result:
        actual = validated.loc[validated.Year == single_result["Year"], "Value"]
        if len(actual):
            st.success(f"Year {single_result['Year']}: fitted estimate = {single_result['Estimate']:.8g} {unit}; actual observed value = {actual.iloc[0]:.8g} {unit}. These are different quantities.")
        else:
            st.success(f"Year {single_result['Year']}: estimate = {single_result['Estimate']:.8g} {unit} ({single_result['Classification']}).")
        if single_result["Classification"].startswith("outside"):
            st.warning("This is polynomial extrapolation outside the fitted historical range. It is not a verified forecast.")
    batch_error = None
    if batch_mode == "Year range":
        if batch_end < batch_start:
            batch_error = "End year must be greater than or equal to start year."
            years_req = []
        else:
            years_req = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
    else:
        years_req = []
        for token in custom_text.split(","):
            token = token.strip()
            if not token: continue
            if not re.fullmatch(r"\d{4}", token):
                batch_error = f"Invalid year '{token}'. Enter comma-separated four-digit whole years."
                continue
            y = int(token)
            if not 1950 <= y <= 2100:
                batch_error = f"Year {y} is outside the allowed 1950–2100 range."
                continue
            years_req.append(y)
        years_req = sorted(set(years_req))
    if batch_error:
        st.error(batch_error)
    batch_df = pd.DataFrame(columns=["Year", "Estimate", "Classification"])
    if years_req:
        batch_vals = model_predict(model, years_req)
        batch_df = pd.DataFrame({"Year": years_req, "Estimate": batch_vals,
            "Classification": ["within fitted historical range" if model["years"].min() <= y <= model["years"].max() else "outside fitted historical range" for y in years_req]})
        batch_df["Actual observation"] = [float(validated.loc[validated.Year == y, "Value"].iloc[0]) if (validated.Year == y).any() else np.nan for y in years_req]
        st.dataframe(batch_df, use_container_width=True)
        st.caption("Batch estimates use exactly the same centered-time polynomial as the single-year calculator.")
    else:
        st.caption("Enter a valid batch range or list of years to calculate a batch.")

    st.subheader("Country comparison")
    compare_labels = st.multiselect("Compare same indicator in other countries", [c for c in COUNTRIES if c != country_label])
    comparisons = []
    for label in compare_labels:
        frame, msg = fetch_world_bank(COUNTRIES[label], indicator_code, start_year, end_year)
        if msg: st.warning(f"{label}: {msg}")
        elif not frame.empty:
            comparisons.append((label, frame))
            st.caption(f"{label}: coverage {frame.Year.min()}–{frame.Year.max()}, {len(frame)} observations. Different coverage and missing years mean comparisons may not be year-for-year complete.")
    fig = make_chart(validated, model, sample_step, horizon,
                     single_result["Year"] if single_result else None,
                     single_result["Estimate"] if single_result else None,
                     batch_df if not batch_df.empty else None, comparisons,
                     "Year", f"{indicator_name} ({unit})")
    st.pyplot(fig, use_container_width=True)
    chart_buffer = io.BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
    chart_bytes = chart_buffer.getvalue()
    plt.close(fig)

    st.subheader("Mathematics pathway")
    if pathway == "Honors Algebra 2":
        st.write(f"- **Domain shown:** {int(validated.Year.min())}–{int(validated.Year.max())} for observed evidence; the polynomial itself is mathematically defined for all real inputs.")
        st.write(f"- **Range on historical observations:** {validated.Value.min():.6g} to {validated.Value.max():.6g} {unit}.")
        st.write("- **End behavior:** determined by the leading coefficient and degree; this mathematical behavior should not be assumed realistic for distant future years.")
        st.write("- **Trend and average rate:** compare two years with (f(b) − f(a))/(b − a); the result is the average annual change.")
    elif pathway == "Precalculus":
        st.write("- The degree controls the maximum possible number of turning points (at most degree − 1); coefficients affect steepness, curvature, and vertical scale.")
        st.write("- Turning points and concavity can be explored in Calculus mode. Here, compare average rates over consecutive intervals: changing average rates indicate changing curvature.")
        st.write(f"- Historical range of observed values: {validated.Value.min():.6g} to {validated.Value.max():.6g} {unit}.")
    else:
        st.write("Calculus mode is available in the Function Studio below, where the selected function, first derivative, second derivative, secant/tangent lines, critical points, extrema, concavity, and inflection tests are analyzed.")

    st.subheader("Function Studio — contexts beyond the historical dataset")
    context = st.selectbox("Application context", list(CONTEXTS))
    dep_default, dep_unit_default, indep_default, indep_unit_default, context_desc, suggested = CONTEXTS[context]
    context_description = st.text_input("Context description", context_desc)
    d1, d2, d3, d4 = st.columns(4)
    with d1: dep_name = st.text_input("Dependent variable", dep_default)
    with d2: dep_unit = st.text_input("Dependent units", dep_unit_default)
    with d3: indep_name = st.text_input("Independent variable", indep_default)
    with d4: indep_unit = st.text_input("Independent units", indep_unit_default)
    function_source = st.radio("Function source", ["Generated historical curve", "Suggested function", "Build with parameters", "Enter my own function"], horizontal=False)
    suggested_type = st.selectbox("Suggested function family", FUNCTION_TYPES)
    param_a = st.number_input("Parameter a", value=2.0)
    param_b = st.number_input("Parameter b", value=0.15)
    param_c = st.number_input("Parameter c", value=1.0)
    custom_expression = st.text_input("Function in x (use ^ or ** for powers)", "2*sin(0.5*x)+3")
    if function_source == "Generated historical curve":
        x_sym = sp.symbols("x", real=True)
        t_sym = sp.symbols("t", real=True)
        expr = sum(sp.Float(float(c)) * (x_sym - model["base"])**(degree-i) for i, c in enumerate(model["coeffs"]))
        studio_label = f"Historical polynomial shifted by base year {model['base']}"
    elif function_source == "Suggested function":
        suggested_map = {
            "Polynomial": f"{param_a}*x**2+{param_b}*x+{param_c}",
            "Rational": f"{param_a}/(x-{param_b})+{param_c}",
            "Exponential": f"{param_a}*exp({param_b}*x)+{param_c}",
            "Logarithmic": f"{param_a}*log(x)+{param_c}",
            "Trigonometric": f"{param_a}*sin({param_b}*x)+{param_c}",
        }
        x_sym, expr = safe_expression(suggested_map[suggested_type])
        studio_label = f"Suggested {suggested_type.lower()} function"
    elif function_source == "Build with parameters":
        source = f"{param_a}*x**2+{param_b}*x+{param_c}"
        x_sym, expr = safe_expression(source)
        studio_label = "Parameterized quadratic"
    else:
        try:
            x_sym, expr = safe_expression(custom_expression)
            studio_label = "User-entered function"
        except ValueError as exc:
            st.error(str(exc))
            x_sym, expr = sp.symbols("x", real=True), sp.Integer(0)
            studio_label = "Invalid input replaced by zero only for safe display"
    if function_source == "Generated historical curve" and not validated.empty:
        default_low = float(validated["Year"].min())
        default_high = float(validated["Year"].max())
        if default_high <= default_low:
            default_high = default_low + 1.0
        indep_name = st.text_input("Historical independent variable label", "Year", key="historical_x_label")
        indep_unit = st.text_input("Historical independent variable units", "calendar year", key="historical_x_unit")
        dep_name = st.text_input("Historical dependent variable label", indicator_name, key="historical_y_label")
        dep_unit = st.text_input("Historical dependent variable units", unit, key="historical_y_unit")
    else:
        default_low, default_high = 0.1, 10.0
    interval_cols = st.columns(2)
    with interval_cols[0]: interval_low = st.number_input("Analysis interval start", value=float(default_low))
    with interval_cols[1]: interval_high = st.number_input("Analysis interval end", value=float(default_high))
    if interval_high <= interval_low:
        st.error("Analysis interval end must be greater than start.")
        interval_low, interval_high = default_low, default_high
    derivative = sp.diff(expr, x_sym)
    second_derivative = sp.diff(derivative, x_sym)
    st.markdown(f"**{studio_label}:**")
    st.latex(sp.latex(expr))
    if pathway == "Calculus":
        st.markdown("**First derivative**")
        st.latex(sp.latex(derivative))
        st.markdown("**Second derivative**")
        st.latex(sp.latex(second_derivative))
    show_f = st.checkbox("Show function f(x)", True)
    show_d1 = st.checkbox("Show first derivative f′(x)", pathway == "Calculus")
    show_d2 = st.checkbox("Show second derivative f″(x)", False)
    point_x = st.slider("Current input x", float(interval_low), float(interval_high),
                        float((interval_low+interval_high)/2), step=max((interval_high-interval_low)/200, .001))
    s1, s2 = st.columns(2)
    with s1:
        sec_a = st.slider("Secant interval start a", float(interval_low), float(interval_high), float(interval_low))
    with s2:
        sec_b = st.slider("Secant interval end b", float(interval_low), float(interval_high), float(interval_high))
    play_cols = st.columns(4)
    with play_cols[0]:
        if st.button("▶ Play one step"):
            st.session_state["animation_x"] = min(interval_high, point_x + (interval_high-interval_low)/20)
            st.info(f"Next animation position: {st.session_state['animation_x']:.4g}. Move the current-input slider to synchronize the view.")
    with play_cols[1]:
        if st.button("⏸ Pause"):
            st.session_state["animation_paused"] = True
            st.caption("Playback paused.")
    with play_cols[2]:
        if st.button("↺ Reset"):
            st.session_state["animation_x"] = interval_low + (interval_high-interval_low)/2
            st.caption("Animation position reset; set the slider to the displayed position.")
    with play_cols[3]:
        speed = st.selectbox("Playback step", ["Slow", "Normal", "Fast"], index=1)
    current_y = evaluate_scalar(expr, x_sym, point_x)
    current_d = evaluate_scalar(derivative, x_sym, point_x)
    current_dd = evaluate_scalar(second_derivative, x_sym, point_x)
    st.write(f"**At x = {point_x:.5g} {indep_unit}:** f(x) = {fmt_num(current_y)} {dep_unit}; f′(x) = {fmt_num(current_d)} {dep_unit}/{indep_unit}; f″(x) = {fmt_num(current_dd)} {dep_unit}/{indep_unit}².")
    st.write(f"In context: {context_description}. The function gives {dep_name}; its first derivative describes the instantaneous rate of change of {dep_name} per {indep_name}; its second derivative describes how that rate is changing.")
    if sec_b != sec_a:
        ya, yb = evaluate_scalar(expr, x_sym, sec_a), evaluate_scalar(expr, x_sym, sec_b)
        if ya is not None and yb is not None:
            avg_rate = (yb-ya)/(sec_b-sec_a)
            st.write(f"**Average rate of change** from {sec_a:.4g} to {sec_b:.4g}: {avg_rate:.7g} {dep_unit}/{indep_unit}.")
        else:
            st.warning("The function is undefined at one or both secant endpoints.")
    else:
        st.caption("Choose different secant endpoints to calculate average rate of change.")
    if current_y is not None and current_d is not None:
        st.write(f"**Tangent line at x = {point_x:.4g}:** y = {current_y:.6g} + ({current_d:.6g})(x − {point_x:.6g}).")
    else:
        st.warning("The function or derivative is undefined at the selected point; no tangent line is displayed.")

    # Numerical calculus analysis: approximate candidates are checked with sign changes.
    try:
        f_func = sp.lambdify(x_sym, expr, modules=["numpy"])
        d_func = sp.lambdify(x_sym, derivative, modules=["numpy"])
        dd_func = sp.lambdify(x_sym, second_derivative, modules=["numpy"])
        grid = np.linspace(interval_low, interval_high, 1200)
        def vector_eval(fn, xx):
            with np.errstate(all="ignore"):
                yy = np.asarray(fn(xx), dtype=float)
                if yy.ndim == 0: yy = np.full_like(xx, float(yy))
                yy[~np.isfinite(yy)] = np.nan
                return yy
        fy, dy, ddy = vector_eval(f_func, grid), vector_eval(d_func, grid), vector_eval(dd_func, grid)
        critical = numerical_roots(derivative, x_sym, interval_low, interval_high)
        stationary_inflections = numerical_roots(second_derivative, x_sym, interval_low, interval_high)
        st.markdown("**Calculus analysis (numerical candidates; values are approximations)**")
        if pathway == "Calculus":
            classified = []
            for root in critical:
                left, right = max(interval_low, root-1e-3), min(interval_high, root+1e-3)
                dl, dr = evaluate_scalar(derivative, x_sym, left), evaluate_scalar(derivative, x_sym, right)
                kind = "local minimum" if dl is not None and dr is not None and dl < 0 < dr else "local maximum" if dl is not None and dr is not None and dl > 0 > dr else "stationary point (not classified as extremum)"
                classified.append((root, evaluate_scalar(expr, x_sym, root), kind))
            st.write("Critical points:", classified if classified else "No numerical critical points found in this interval.")
            st.write("Increasing/decreasing behavior is determined by the sign of f′(x) between critical points; undefined/discontinuous regions must be treated separately.")
            infl = []
            for root in stationary_inflections:
                left, right = max(interval_low, root-1e-3), min(interval_high, root+1e-3)
                vl, vr = evaluate_scalar(second_derivative, x_sym, left), evaluate_scalar(second_derivative, x_sym, right)
                if vl is not None and vr is not None and vl*vr < 0:
                    infl.append((root, evaluate_scalar(expr, x_sym, root), "concavity changes; numerical inflection candidate"))
            st.write("Verified numerical inflection candidates (concavity sign changes):", infl if infl else "No concavity-changing candidate found.")
            st.write(f"Zeros of f″(x) tested: {stationary_inflections}. A zero alone is not sufficient to establish an inflection point.")
            endpoints = [(interval_low, evaluate_scalar(expr, x_sym, interval_low)), (interval_high, evaluate_scalar(expr, x_sym, interval_high))]
            finite_values = [(xx, yy) for xx, yy in endpoints if yy is not None]
            finite_values += [(xx, yy) for xx, yy, _ in classified if yy is not None]
            if finite_values:
                st.write("Endpoint/stationary-point candidates for absolute extrema on the closed interval:", {
                    "minimum": min(finite_values, key=lambda z:z[1]),
                    "maximum": max(finite_values, key=lambda z:z[1])})
            st.write("Domain/discontinuities: inspect denominator zeros, logarithm inputs (must be positive), and square-root inputs (must be nonnegative). Numerical graph gaps mark undefined samples.")
        elif pathway == "Precalculus":
            st.write(f"Approximate turning-point candidates from f′(x): {critical or 'none detected'}. Compare average rates and the sign/shape of the second derivative to discuss concavity.")
        else:
            st.write(f"Approximate stationary candidates: {critical or 'none detected'}. Use graph shape and changing average rates to discuss trends and end behavior.")

        fig2, ax2 = plt.subplots(figsize=(10, 4.8))
        if show_f: ax2.plot(grid, fy, label="f(x)", linewidth=2)
        if show_d1: ax2.plot(grid, dy, label="f′(x)", alpha=.8)
        if show_d2: ax2.plot(grid, ddy, label="f″(x)", alpha=.8)
        if current_y is not None: ax2.scatter([point_x], [current_y], color="red", zorder=5, label="Current point")
        if current_y is not None and current_d is not None:
            tangent_x = np.linspace(max(interval_low, point_x-(interval_high-interval_low)/5), min(interval_high, point_x+(interval_high-interval_low)/5), 80)
            ax2.plot(tangent_x, current_y+current_d*(tangent_x-point_x), "--", label="Tangent line")
        if sec_b != sec_a:
            ya, yb = evaluate_scalar(expr, x_sym, sec_a), evaluate_scalar(expr, x_sym, sec_b)
            if ya is not None and yb is not None:
                slope = (yb-ya)/(sec_b-sec_a)
                sec_x = np.linspace(interval_low, interval_high, 100)
                ax2.plot(sec_x, ya+slope*(sec_x-sec_a), ":", label="Secant line")
                ax2.scatter([sec_a, sec_b], [ya, yb], s=22, label="Secant endpoints")
        ax2.axhline(0, color="gray", linewidth=.6)
        ax2.set_xlabel(f"{indep_name} ({indep_unit})")
        ax2.set_ylabel("Function / derivative values (respective units)")
        ax2.set_title("Function and derivative explorer")
        ax2.grid(True, alpha=.25); ax2.legend(); fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        sim = st.selectbox("Contextual visual simulation", SIMULATIONS)
        st.caption(f"Simulation: {sim}. The current input x = {point_x:.4g} and f(x) = {fmt_num(current_y)} are the model state. Visual scaling is conceptual, not a physical calibration; use the stated units for numerical interpretation.")
        if current_y is not None:
            normalized = 0.5 if not np.isfinite(current_y) else 0.5 + 0.45*np.tanh(current_y/10)
            st.progress(float(np.clip(normalized, 0, 1)), text=f"Scaled indicator: {normalized:.1%} of display gauge (nonphysical visualization scale)")
    except Exception as exc:
        st.warning(f"Could not complete numerical analysis for this expression/interval: {exc}")

    summary = {
        "Application": APP_TITLE, "Research question": research_question, "Country": country_label,
        "Indicator": indicator_name, "Indicator code": indicator_code, "Units": unit,
        "Source": source_url, "Retrieved": datetime.now().astimezone().isoformat(timespec="minutes"),
        "Actual coverage": f"{int(validated.Year.min())}–{int(validated.Year.max())}",
        "Observation count": len(validated), "Data status": "User-modified scenario" if is_edited else "Source data",
        "Degree": degree, "Base year": model["base"], "Equation": equation,
        "In-sample R-squared": model["r2"], "In-sample MAE": model["mae"],
        "Mean": model["mean"], "Pathway": pathway,
        "Limitations": "Polynomial extrapolation is uncertain; MAE is not a prediction interval; correlation/trend does not establish historical cause."
    }
    st.subheader("Exports")
    csv_bytes = validated.to_csv(index=False).encode("utf-8")
    xlsx_bytes = build_excel(summary, validated, batch_df)
    pdf_bytes = build_pdf(summary, chart_bytes, validated.head(30), batch_df)
    e1, e2, e3 = st.columns(3)
    with e1: st.download_button("Download working data CSV", csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
    with e2: st.download_button("Download Excel workbook", xlsx_bytes, file_name="hispanic_heritage_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with e3: st.download_button("Download PDF report", pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
    if printer_mode:
        st.divider()
        st.header("Printer-friendly report")
        st.write(f"**{country_label} · {indicator_name} · {unit}**")
        st.write(f"Coverage {summary['Actual coverage']} ({len(validated)} observations); degree {degree}; {summary['Data status']}.")
        st.write(equation)
        st.write(f"In-sample R² {model['r2']:.5g}; MAE {model['mae']:.5g} {unit}.")
        st.write("Findings should describe the observed trend, not claim causes. Extrapolated polynomial values are uncertain.")
        st.pyplot(make_chart(validated, model, sample_step, horizon, x_label="Year", y_label=f"{indicator_name} ({unit})"), use_container_width=True)

    with st.expander("How the app works and mathematical assumptions"):
        st.markdown("""
        1. **Retrieve:** a cached request reads annual observations from the World Bank API. Null years are omitted, never fabricated.
        2. **Validate/edit:** source data remains separate from the editable working table. Years must be unique integers and values must satisfy basic indicator constraints.
        3. **Fit:** the polynomial uses centered time, `t = year − base year`, to reduce numerical conditioning problems. Degree `d` requires at least `d+1` distinct years.
        4. **Evaluate:** all single-year and batch predictions call the same fitted polynomial with the same centering transformation.
        5. **Interpret:** in-sample R² and MAE describe fit to the supplied observations; a chronological holdout score is shown separately when there is enough data.
        6. **Limitations:** a polynomial is descriptive, sensitive to degree/outliers, and often unreliable outside the observed range. It does not establish historical causes. No inferential prediction interval is claimed.
        7. **Function Studio:** SymPy parses a restricted set of familiar functions and differentiates symbolically. Root finding and sign-change classification are numerical approximations over the selected interval, not exact proofs.
        """)

if __name__ == "__main__":
    main()
