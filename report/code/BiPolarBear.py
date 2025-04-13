import polars as pl
from AI_models import YogobellaMLLMix
YogobellaMLLMix = YogobellaMLLMix()


class BiPolarBear:
    def __init__(self):
        #self.df = pl.read_csv("report\data\RTVSlo\PrometnoPorocilo2022.csv", encoding="ANSI")
        self.olamma = YogobellaMLLMix.ollama_model(model_name="deepseek-r1")
        self.df = pl.read_csv("report\data\RTVSlo\PrometnoPorocilo2022.csv", encoding="ANSI")
        self.sample_df = pl.read_csv("report\data\RTVSlo\PrometnoPorocilo2022_sample.csv", encoding="ANSI")

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

    def print_important_news(self, columns= ["TitlePomembnoSLO", "ContentPomembnoSLO"]):
        # Extract the relevant columns from the DataFrame
        important_news_df = self.df.select(columns)

        #get only unique content rows
        important_news_df = important_news_df.unique(subset="ContentPomembnoSLO")
        # Iterate through each row in the DataFrame and print the news
        for index, row in enumerate(important_news_df.iter_rows()):
            if row[1] is None:
                continue
            print(f"Title: {row[0]}")
            print(f"Content: {row[1]}")
            print("-" * 80)

    def find_match(self, date):
        """
        Find and match data from PrometnoPorocilo_2022 to the appropriate RTF file based on date.
        
        Args:
            date (str): Date in format 'YYYY-MM-DD' to match
            
        Returns:
            tuple: (rtf_content, matching_data) where rtf_content is the content of the RTF file
                  and matching_data is the corresponding row from the DataFrame
        """
        # Convert date to match the format in the DataFrame
        # Assuming the date column is named 'Date' in the DataFrame
        matching_rows = self.df.filter(pl.col('Date') == date)
        
        if len(matching_rows) == 0:
            raise ValueError(f"No data found for date {date}")
            
        # Get the matching data
        matching_data = matching_rows.to_dicts()[0]
        
        # Construct the RTF filename based on the date
        rtf_filename = f"report/data/RTVSlo/rtf_files/{date}.rtf"
        
        try:
            with open(rtf_filename, 'r', encoding='utf-8') as file:
                rtf_content = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"No RTF file found for date {date}")
            
        return rtf_content, matching_data

    def process_rtf_files(self):
        """
        Process all RTF files in the Podatki - rtvslo.si folder, extract date, time and content,
        and save the information to a joined CSV file.
        """
        import os
        import re
        from datetime import datetime
        
        # Initialize lists to store data
        all_data = []
        
        # Walk through the directory
        base_path = "report/data/RTVSlo/Podatki - rtvslo.si"
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.rtf'):
                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    
                    try:
                        # Read file with ANSI encoding
                        with open(file_path, 'r', encoding='ANSI') as f:
                            content = f.read()
                            
                            # Extract date and time using regex
                            date_time_match = re.search(r'(\d{1,2}\.\s+\d{1,2}\.\s+\d{4})\s+(\d{1,2}\.\d{2})', content)
                            if date_time_match:
                                date_str = date_time_match.group(1)
                                time_str = date_time_match.group(2)
                                
                                # Convert date to standard format
                                date_obj = datetime.strptime(date_str, '%d. %m. %Y')
                                formatted_date = date_obj.strftime('%d. %m. %Y')
                                
                                # Extract content sections
                                content_sections = []
                                # Look for content after "Podatki o prometu." or similar headers
                                content_start = re.search(r'Podatki o prometu\.', content)
                                if content_start:
                                    remaining_content = content[content_start.end():]
                                    # Split content into sections (assuming they're separated by \par)
                                    sections = re.split(r'\\par\s*\\par', remaining_content)
                                    for section in sections:
                                        # Clean up the content while preserving special characters
                                        clean_section = section
                                        # Remove RTF commands but keep special character codes
                                        clean_section = re.sub(r'\\[a-zA-Z0-9]+(?![0-9])', '', clean_section)
                                        # Remove RTF groups
                                        clean_section = re.sub(r'\{.*?\}', '', clean_section)
                                        # Convert RTF special character codes to actual characters
                                        clean_section = re.sub(r'\\\'([0-9a-fA-F]{2})', 
                                                             lambda m: chr(int(m.group(1), 16)), 
                                                             clean_section)
                                        clean_section = clean_section.strip()
                                        if clean_section:
                                            content_sections.append(clean_section)
                                
                                # Create row data
                                row_data = {
                                    'Datum': formatted_date,
                                    'ura': time_str,
                                    'TMP_file_name': file,
                                    'TMP_folder_name': folder_name
                                }
                                
                                # Add content sections
                                for i, section in enumerate(content_sections, 1):
                                    row_data[f'content_{i:02d}'] = section
                                
                                all_data.append(row_data)
                                
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
        
        # Convert to DataFrame
        df = pl.DataFrame(all_data)
        
        # Save to CSV with proper encoding
        output_path = "report/data/RTVSlo/Joined_rtf_files.csv"
        df.write_csv(output_path)
        print(f"Processed {len(all_data)} RTF files and saved to {output_path}")
        return df

if __name__ == "__main__":
    # Example usage
    bipolar_bear = BiPolarBear()
    #bipolar_bear.save_sample(100)  # Save a sample of 100 rows
    #bipolar_bear.inspect()
    #bipolar_bear.print_important_news()  # Print important news
    bipolar_bear.process_rtf_files()  # Process RTF files