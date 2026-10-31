# Sources and indicator definitions

## Primary source

All historical country-indicator observations are requested from the World Bank's official World Development Indicators API:

- API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
- API endpoint pattern: `https://api.worldbank.org/v2/country/{ISO3}/indicator/{INDICATOR_CODE}?format=json&date=START:END&per_page=20000`
- Country indicator pages: `https://data.worldbank.org/indicator/{INDICATOR_CODE}?locations={ISO3}`

The application requests the 70 completed calendar years from current year minus 70 through current year minus 1. It reports actual non-null coverage and count. Missing observations remain missing; no random or synthetic historical data are inserted.

## Country codes

| Country | World Bank API code |
|---|---|
| Mexico | MEX |
| Brazil | BRA |
| Argentina | ARG |
| Colombia | COL |
| Chile | CHL |
| Peru | PER |
| Venezuela | VEN |
| Ecuador | ECU |
| Uruguay | URY |

Brazil is included as a Latin American comparison country. It is not a Spanish-speaking country.

## Indicators

| Indicator | World Bank code | Unit | Definition / interpretation |
|---|---|---|---|
| Life expectancy at birth | `SP.DYN.LE00.IN` | years | Expected years a newborn would live if current age-specific mortality rates remained constant. |
| Unemployment, national estimate | `SL.UEM.TOTL.NE.ZS` | % of labor force | Share of the labor force without work but available for and seeking employment, using national estimates. |
| GDP per capita | `NY.GDP.PCAP.CD` | current US$ per person | GDP divided by midyear population. It is not average wealth or household income. |
| Crude birth rate | `SP.DYN.CBRT.IN` | births per 1,000 people per year | Annual live births per 1,000 population; it is a rate, not a count. |
| Net migration | `SM.POP.NETM` | people | Net migrants (immigrants minus emigrants); it is not an emigration count. |
| Intentional homicide rate | `VC.IHR.PSRC.P5` | deaths per 100,000 people | Intentional homicide victims per 100,000 population. |
| Consumer price inflation | `FP.CPI.TOTL.ZG` | % annual | Annual percentage change in the consumer price index. |
| Central government debt | `GC.DOD.TOTL.GD.ZS` | % of GDP | Central government debt as a percentage of GDP. |
| Total reserves including gold | `FI.RES.TOTL.CD` | current US$ | Total reserves including gold, valued in current US dollars. |
| External balance on goods and services | `NE.RSB.GNFS.ZS` | % of GDP | External balance on goods and services as a percentage of GDP. |

## Retrieval date

The app displays the retrieval date/time at runtime. The World Bank may revise historical observations; exported reports should be treated as snapshots of the values returned on their retrieval date.

## Important caveats

- Indicator definitions, coverage, methodology, and availability can vary over time and across countries.
- The API may return nulls or no observations for some country-indicator-year combinations.
- National-estimate unemployment is not interchangeable with modeled ILO estimates.
- GDP per capita is an aggregate ratio and does not describe how income is distributed.
- Net migration is a net balance, not the number of people leaving a country.
- Historical association and polynomial trends do not establish historical causes.
