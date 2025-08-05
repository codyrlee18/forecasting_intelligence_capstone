import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

def rev_analysis():

    df = pd.read_csv("/opt/airflow/clean_data/revenue_forecast_data.csv")

    df = df[df['deal_status'] == 'Closed -Won']
        
    df['month'] = pd.to_datetime(df['month'])

    fc_df = df.groupby('month')['actual_usd'].sum().reset_index()
    import seaborn as sb
    fc_df['month'] = pd.to_datetime(fc_df['month'])

    sb.lineplot(x = 'month', y = 'actual_usd', data=fc_df)
    plot_acf(fc_df.actual_usd)

    # Run ADF test      
    result = adfuller(fc_df.actual_usd.dropna())

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])

    # We can therefore assume the series is stationary, and therefore usable to model with ARIMA.
    model = ARIMA(fc_df.actual_usd, order=(2,1,0))

    model = model.fit()

    print(model.summary())
    model.plot_diagnostics(figsize=(10, 6))


    forecast_steps = 6
    forecast = model.get_forecast(steps=forecast_steps)
    conf_int = forecast.conf_int()
    forecast_mean = forecast.predicted_mean
    fc_df.set_index('month', inplace=True)


    # Create future forecast index (monthly steps after last date)
    forecast_index = pd.date_range(start=fc_df.index[-1] + pd.DateOffset(months=1),
                                periods=forecast_steps, freq='M')

    # Set the index for forecast results
    forecast_mean.index = forecast_index
    conf_int.index = forecast_index

    def millions_formatter(x, pos):
        """Converts a number to millions with 'M' suffix."""
        return '{:.0f}M'.format(x * 1e-6)

    # 5. Plot the original series and the forecast
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    plt.plot(fc_df.index, fc_df.actual_usd, label='Historical Data')
    plt.plot(forecast_mean, label='Forecasted Revenue', linestyle='--', color='green')
    plt.fill_between(forecast_mean.index, 
                    conf_int.iloc[:, 0], 
                    conf_int.iloc[:, 1], 
                    color='lightblue', alpha=0.2, label='95% CI')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))
    plt.title('Actual Revenue and Projected Six-Month')
    plt.xlabel('Date (Year-Month)')
    plt.ylabel('$USD (Millions)')
    plt.legend()
    plt.grid(False)
    plt.savefig("/opt/airflow/outputs/forecast_plot.png", transparent= True)

    formatted_projection = forecast_mean.reset_index()
    formatted_projection.columns = ['Month', 'Projected Revenue']
    formatted_projection['Projected Revenue'] = formatted_projection['Projected Revenue'].apply(lambda x:f"${x:,.2f}")
    output_file_path_forecast = '/opt/airflow//outputs/forecast_table.csv'
    formatted_projection.to_csv(output_file_path_forecast, index=False)
    print(f"Forecast table saved to {output_file_path_forecast}")

if __name__ == "__main__":
    rev_analysis()