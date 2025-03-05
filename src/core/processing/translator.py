# src/core/processing/translator.py
import pandas as pd
from deep_translator import GoogleTranslator
import time

class Translator:
    def __init__(self, source='en', target='ko'):
        self.source = source
        self.target = target

    def translate_text(self, text):
        """Translate a single text string from source to target language."""
        if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
            return text
        try:
            translated = GoogleTranslator(source=self.source, target=self.target).translate(text)
            return translated
        except Exception as e:
            print(f"Error translating text: {e}")
            return text

    def translate_dataframe_column(self, df, src_column, dest_column, sleep_time=0.3):
        """
        Translate all entries in a specified column of a DataFrame.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the text.
            src_column (str): The name of the column with the original text.
            dest_column (str): The name for the new column to store translations.
            sleep_time (float): Pause time between translations (to avoid rate limits).
            
        Returns:
            pd.DataFrame: The updated DataFrame with a new translation column.
        """
        translations = []
        for text in df[src_column]:
            translation = self.translate_text(text)
            translations.append(translation)
            time.sleep(sleep_time)
        df[dest_column] = translations
        return df