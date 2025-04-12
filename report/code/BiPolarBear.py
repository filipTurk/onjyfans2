import polars as pl

class BiPolarBear:
    def __init__(self):
        #self.df = pl.read_csv("report\data\RTVSlo\PrometnoPorocilo2022.csv", encoding="ANSI")
        self.df = pl.read_csv("report\data\RTVSlo\PrometnoPorocilo2022_sample.csv", encoding="ANSI")

    def save_sample(self, sample_size=100):
        sample = self.df.sample(sample_size, with_replacement=False)
        sample.write_csv("report\data\RTVSlo\PrometnoPorocilo2022_sample.csv")

    def inspect(self):
        print(self.df.head())
        for column in self.df.columns:
            print(f"Column: {column}, Type: {self.df[column].dtype}")

        columns = self.df.columns  # Save column order for indexing
        for index, row in enumerate(self.df.iter_rows()):
            if index < 5:  # Limit to first 5 rows for inspection
                for col_idx, column in enumerate(columns):
                    print(f"Row {index}, Column {column}: {row[col_idx]}")
            else:
                break

        

if __name__ == "__main__":
    # Example usage
    bipolar_bear = BiPolarBear()
    bipolar_bear.inspect()