import glob
from langchain_community.document_loaders import Docx2txtLoader


class DocumentLoader:
    def __init__(self, folder_path: str):
        """
        Initialize the DocumentLoader with the path to the folder containing
        .docx files.
        :param folder_path: Path to the folder containing .docx files.
        """
        self.folder_path = folder_path
        self.filenames = self._get_all_docx_files()
        self.data = []

    def _get_all_docx_files(self):
        """
        Retrieve all .docx files in the specified folder.
        :return: List of paths to .docx files.
        """
        return glob.glob(self.folder_path + '*.docx')

    def load_documents(self):
        """
        Load all .docx files and return the data.
        :return: List containing the loaded documents.
        """
        i = 0
        for filename in self.filenames:
            i += 1
            try:
                loader = Docx2txtLoader(filename)
                self.data.extend(loader.load())
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        print(f"Total documents loaded: {len(self.data)}")
        return self.data
