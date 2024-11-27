import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def get_euribor_annual_rates(years_to_fetch=10):
    """
    Collects the annual Euribor rates for the months of June (6) and November (11)
    from the last 'years_to_fetch' years, including the current year.

    Args:
        years_to_fetch (int): Number of years to fetch, including the current year.

    Returns:
        DataFrame: Euribor rates organized with columns for Year, Month, and Annual Rate.
    """
    # Declare the global variable before any assignment
    global df_euribor

    base_url = "https://www.euribor-rates.eu/en/euribor-rates-by-year/"
    all_data = []

    # Determine the current year and the previous years
    current_year = datetime.now().year
    years = range(current_year - years_to_fetch + 1, current_year + 1)

    for year in years:
        url = f"{base_url}{year}/"
        print(f"Fetching data for the year {year}...")
        response = requests.get(url)

        # Check if the page loaded correctly
        if response.status_code != 200:
            print(f"Error accessing {url}. Skipping to the next year.")
            continue

        # Parse the page
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the rates table
        table = soup.find("table", class_="table table-striped")
        if not table:
            print(f"Rates table not found for the year {year}. Skipping...")
            continue

        # Process the table rows
        rows = table.find("tbody").find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            date_col = row.find("th")  # Date is in the first cell of the row (<th>)
            if not date_col or len(cols) < 5:  # Check if there is enough data
                continue

            # Extract date and annual rate
            date = date_col.text.strip()  # Example: "11/1/2024"
            annual_rate = cols[-1].text.strip().replace("%", "")  # Last cell

            # Filter for the months of June (6) and November (11)
            if date.startswith("6/") or date.startswith("11/"):
                month = int(date.split("/")[0])  # Extract the month
                data_entry = {
                    "Year": year,
                    "Month": month,
                    "Annual Euribor Rate": float(annual_rate)
                }
                all_data.append(data_entry)

    # Create a DataFrame and sort it in reverse chronological order
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values(by=["Year", "Month"], ascending=[False, False]).reset_index(drop=True)

        # Generate the CSV file name
        filename = f"euribor_annual_rates_{datetime.now().year - 9}_{datetime.now().year}.csv"
        filepath = os.path.join(os.getcwd(), filename)

        # Save the DataFrame as a CSV file
        df.to_csv(filepath, index=False)

        # Assign to the global environment
        df_euribor = df

        # Inform the user about the saved file and the DataFrame
        print(f"\nThe generated DataFrame was saved as 'df_euribor'")
        print(f"The CSV file was saved at: #filepath")
        print(f"\nDataFrame successfully generated! Displaying first rows:\n")
        print(df.head())

        # Return the DataFrame
        return df
    else:
        print("No data was collected. The CSV file was not generated.")
        df_euribor = pd.DataFrame()
        return df_euribor

# Protected main block
if __name__ == "__main__":
    print("Starting data collection...")
    df_euribor = get_euribor_annual_rates(years_to_fetch=10)



