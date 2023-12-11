import os
import base64
import time
import streamlit as st
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from typing import Optional
import pandas as pd


class ReportGenerator:

    def __init__(self):
        self.index_loader = IndexLoader()  # Class instance for index related tasks
        self.index = None

    def read_and_encode_file(self, bin_file: str) -> Optional[str]:
        try:
            with open(bin_file, 'rb') as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except FileNotFoundError as e:
            st.error(f"File {bin_file} not found!: {str(e)}")
            return None

    def get_binary_file_downloader_html(self, bin_file: str, file_label: str ='File') -> Optional[str]:
        bin_str = self.read_and_encode_file(bin_file)
        if bin_str is not None:
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
            return href
        return None

    def generate_report(self, query_engine, query: str="""Generate a comprehensive report analyzing the match between an uploaded resume and a given role. The uploaded resume and role information are provided as PDF files.

1. **Resume Analysis:**
   - Parse and analyze the content of the uploaded resume.
   - Identify and highlight key features, qualifications, and experiences mentioned in the resume.
   - Provide an overview of the candidate's skills and expertise.

2. **Role Comparison:**
   - Extract relevant information from the provided role details.
   - Compare the skills, qualifications, and experiences required for the role with those present in the resume.
   - Evaluate the extent to which the candidate matches the expectations of the role.

3. **Strengths and Weaknesses:**
   - Identify and emphasize the candidate's strengths based on the resume analysis.
   - Highlight any potential weaknesses or areas that may need improvement.

4. **Overall Assessment:**
   - Provide a summary of the overall fit between the candidate's profile and the role requirements.
   - Offer insights into how well the candidate aligns with the expectations of the position.

5. **Recommendations:**
   - Suggest any specific recommendations for further development or areas for improvement.
   - Offer guidance on potential next steps in the hiring process.

Please generate a detailed report considering the points mentioned above. If necessary, request additional information from the user to enhance the analysis. Ensure that the report is clear, concise, and provides actionable insights for decision-making in the hiring process.
""") -> Optional[dict]:
        try:
            response = query_engine.query(query)
            return response
        except Exception as e:
            st.error(f"Error occurred while running query or writing response. {e}")
            return None

    def write_report_to_file(self, response: dict, filename: str ='report.txt') -> str:
        try:
            with open(filename, 'w') as f:
                f.write(str(response))
            return filename
        except Exception as e:
            st.error(f'An error occured while writing the report to a file: {str(e)}')
            return None

    def run_app(self):
        st.title('BrightPath: Resume Report Generator with AI')
        role_info = st.file_uploader("Upload the Role Information PDF", type=['pdf'])
        resume = st.file_uploader("Upload the Resume PDF", type=['pdf'])

        self.index = self.index_loader.load_index() if not os.path.exists("./storage") else self.index_loader.load_existing_index()

        if self.index is None:
            st.error("Index could not be loaded.")
            return
        if st.button('Generate Report'):
            if role_info is None or resume is None:
                st.warning("Please upload both Role Information and Resume PDF to generate report!")
                return
            query_engine = self.index.as_query_engine() 
            response = self.generate_report(query_engine)

            if response is not None: 
                report_file = self.write_report_to_file(response)
                file_path = report_file
                with open(file_path, 'r') as file:
                    file_content = file.read()
                st.write(file_content)
                st.markdown(self.get_binary_file_downloader_html(report_file, 'Download Report'), unsafe_allow_html=True)
        time.sleep(1)


class IndexLoader:

    def load_index(self, path="data") -> VectorStoreIndex:
        try:
            documents = SimpleDirectoryReader(path).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist()
            return index
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            return None

    def load_existing_index(self, path="./storage") -> VectorStoreIndex:
        try:
            storage_context = StorageContext.from_defaults(persist_dir=path)
            index = load_index_from_storage(storage_context)
            return index
        except Exception as e:
            st.error(f"Error loading existing index: {str(e)}")
            return None


if __name__ == "__main__": 
    ReportGenerator().run_app()