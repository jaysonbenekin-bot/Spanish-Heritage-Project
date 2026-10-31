# Hispanic Heritage Historical Data Explorer

A self-contained Streamlit app for exploring official World Bank historical indicators and practicing polynomial modeling, function analysis, and calculus.

## Included files

- `app.py` — complete Streamlit application
- `requirements.txt` — minimal, unpinned dependencies
- `sources.md` — data definitions and source references

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
```

The app retrieves data directly from the World Bank API. An internet connection is required for source data. It does not require API credentials.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `sources.md` to the repository root.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **New app**, select the repository and branch, and set the main file path to `app.py`.
5. Deploy. Review the app logs if dependency installation or startup fails.

## Deploy on Render

Create a **Python Web Service** connected to the GitHub repository.

- Build command:

```bash
pip install -r requirements.txt
```

- Start command (single line):

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

Set no secrets unless you add a separate authenticated service later. Do not commit credentials.

## How to use

1. Select a country and World Bank indicator in the sidebar.
2. Review the definition, unit, API source, retrieval date, and actual data coverage.
3. Edit the working Year/Value table if conducting a scenario analysis. Use **Reset to source data** to restore the API data.
4. Choose polynomial degree, graph sample increment, and forward display horizon.
5. Calculate a single year and/or a batch of years. The app uses the same centered-time transformation for both.
6. Add comparison countries for the same indicator.
7. Select a mathematics pathway and explore the Function Studio contexts and functions.
8. Download CSV, Excel, and PDF outputs.

## Verification checklist

- [ ] Confirm the selected indicator code matches `sources.md` and the World Bank page.
- [ ] Confirm source coverage and non-null observation count; do not expect every year to have a value.
- [ ] Edit a value and confirm the user-modified scenario notice appears.
- [ ] Press **Reset to source data** and confirm original source values return.
- [ ] Try duplicate years, fractional years, blank values, and indicator-invalid values.
- [ ] Select a degree requiring more observations than are available; confirm fitting is blocked.
- [ ] Compare the displayed polynomial equation with a hand calculation or an independent numerical tool.
- [ ] Confirm R² and MAE are labeled in-sample; check chronological holdout is separately labeled.
- [ ] Calculate a year using the single-year control and include the same year in batch mode; estimates should match.
- [ ] Try malformed custom years and reversed year ranges; confirm clear errors.
- [ ] Turn comparison countries on and off; coverage notes and plotted series should change.
- [ ] Change sample increment, forward horizon, and printer mode; confirm the chart/report changes.
- [ ] Download all three files and open them with the appropriate application.
- [ ] Test function expressions such as `x^2`, `1/(x-2)`, `exp(-0.2*x)`, `log(x)`, and `sin(x)`.
- [ ] Check rational/logarithmic domain restrictions and confirm graph gaps appear at undefined values.
- [ ] Move the current input and secant endpoints; confirm values and tangent/secant output update.
- [ ] Deploy and repeat the checks in the hosted app.

## Troubleshooting

- **World Bank timeout/network error:** check internet access, retry later, and inspect the Streamlit app logs. The app deliberately does not replace missing data with random values.
- **No observations:** try a different country or indicator; some series have sparse reporting or missing recent years.
- **Polynomial fit error:** lower the degree or use a dataset with at least `degree + 1` distinct valid years.
- **Excel export error:** confirm XlsxWriter installed successfully.
- **PDF export error:** confirm ReportLab installed successfully.
- **App fails at startup:** inspect logs for the first Python traceback, check Python version, and reinstall with `pip install -r requirements.txt`.
- **Custom function parse error:** use `x`, `+ - * / **`, parentheses, `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, and `pi`.

## Important mathematical limitations

The fitted polynomial is a descriptive model, not a causal explanation or guaranteed forecast. High-degree polynomials can oscillate, overfit, and grow implausibly outside the data range. MAE is a descriptive in-sample error and is not a 95% prediction interval. Function Studio calculus root classification is numerical and interval-dependent; discontinuities and closely spaced roots can challenge numerical methods. Exact symbolic derivatives are shown, while critical/inflection candidate locations are numerical approximations.
