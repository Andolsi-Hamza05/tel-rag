from typing import Tuple, Dict, List
from docx import Document as DocxDocument


class DocumentProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.terms_definitions, self.abbreviations_definitions = self._read_docx()

    def _read_docx(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Reads a .docx file and categorizes its content into terms and abbreviations."""
        doc = DocxDocument(self.file_path)
        processing_terms = False
        processing_abbreviations = False
        start = 0
        terms_definitions = {}
        abbreviations_definitions = {}
        for para in doc.paragraphs:
            text = para.text.strip()
            if "References" in text:
                start += 1
            if start >= 2:
                if "Terms and definitions" in text:
                    processing_terms = True
                    processing_abbreviations = False
                elif "Abbreviations" in text:
                    processing_abbreviations = True
                    processing_terms = False
                else:
                    if processing_terms and ':' in text:
                        term, definition = text.split(':', 1)
                        terms_definitions[term.strip()] = definition.strip().rstrip('.')
                    elif processing_abbreviations and '\t' in text:
                        abbreviation, definition = text.split('\t', 1)
                        if len(abbreviation) > 1:
                            abbreviations_definitions[abbreviation.strip()] = definition.strip()
        return terms_definitions, abbreviations_definitions

    def _preprocess(self, text: str, lowercase: bool = True) -> str:
        """Converts text to lowercase and removes punctuation."""
        if lowercase:
            text = text.lower()
        punctuations = '''!()-[]{};:'"\\,<>./?@#$%^&*_~'''
        for char in punctuations:
            text = text.replace(char, '')
        return text

    def _find_and_filter_terms(self, sentence: str) -> Dict[str, str]:
        """Finds terms in the given sentence, case-insensitively,
        and filters out shorter overlapping terms."""
        lowercase_sentence = self._preprocess(sentence, lowercase=True)
        matched_terms = {term: self.terms_definitions[term]
                         for term in self.terms_definitions
                         if self._preprocess(term) in lowercase_sentence}
        final_terms = {}
        for term in matched_terms:
            if not any(term in other and term != other for other in matched_terms):
                final_terms[term] = matched_terms[term]
        return final_terms

    def _find_and_filter_abbreviations(self, sentence: str) -> Dict[str, str]:
        """Finds abbreviations in the given sentence, case-sensitively,
        and filters out shorter overlapping abbreviations."""
        processed_sentence = self._preprocess(sentence, lowercase=False)
        words = processed_sentence.split()
        matched_abbreviations = {word: self.abbreviations_definitions[word]
                                 for word in words if word in self.abbreviations_definitions}
        final_abbreviations = {}
        sorted_abbrs = sorted(matched_abbreviations, key=len, reverse=True)
        for abbr in sorted_abbrs:
            if not any(abbr in other and abbr != other for other in sorted_abbrs):
                final_abbreviations[abbr] = matched_abbreviations[abbr]
        return final_abbreviations

    def find_terms_and_abbreviations_in_sentence(self, sentence: str) -> Tuple[List[str], List[str]]:
        """Finds and filters terms and abbreviations in the given sentence.
           Abbreviations are matched case-sensitively, terms case-insensitively,
           and longer terms are prioritized."""
        matched_terms = self._find_and_filter_terms(sentence)
        matched_abbreviations = self._find_and_filter_abbreviations(sentence)

        formatted_terms = [f"{term}: {definition}" for term, definition in matched_terms.items()]
        formatted_abbreviations = [f"{abbr}: {definition}"
                                   for abbr, definition in matched_abbreviations.items()]

        return formatted_terms, formatted_abbreviations

    def get_definitions(self, sentence: str) -> List[str]:
        """Gets definitions of terms and abbreviations found in the sentence."""
        formatted_terms, formatted_abbreviations = self.find_terms_and_abbreviations_in_sentence(sentence)
        return formatted_terms + formatted_abbreviations

    def define_TA_question(self, sentence: str) -> str:
        """Generates a question including terms and abbreviations found in the sentence."""
        formatted_terms, formatted_abbreviations = self.find_terms_and_abbreviations_in_sentence(sentence)
        terms = '\n'.join(formatted_terms)
        abbreviations = '\n'.join(formatted_abbreviations)
        question = f"""{sentence}\n
Terms and Definitions:\n
{terms}\n
Abbreviations:\n
{abbreviations}\n
"""
        return question
