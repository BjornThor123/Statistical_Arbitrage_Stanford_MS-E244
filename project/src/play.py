from data_loader import DataLoader

loader = DataLoader(data_path = "/Users/kadenwu/Desktop/School/Grad/Year 1/Winter_26/Statistical Arbitrage/Statistical_Arbitrage_Stanford_MS-E244/project/data")
#loader.build_all()
# Get option data from GS ticker and years 2018-2020
query = "SELECT * FROM options WHERE ticker = 'GS' and date >= '2018-01-01' and date <= '2020-12-31'"
query = "SELECT DISTINCT ticker FROM options" 
option_data = loader.query(query)
print(option_data.head())

