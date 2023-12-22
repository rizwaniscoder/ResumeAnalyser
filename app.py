import streamlit as st
import openai
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import PyPDF2
import base64


# Set the page layout to a wider layout
st.set_page_config(layout="wide")

# Ask for OpenAI API key
openai_api_key = st.text_input("Enter your OpenAI API key:")
if not openai_api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

# Set OpenAI API key
openai.api_key = openai_api_key

def analyze_resume(job_desc_text, resume_text, options):
    # Check if job_desc_text is uploaded
    if job_desc_text is not None:
        try:
            # Attempt to decode the job description content as UTF-8
            job_desc = job_desc_text.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            # If decoding as UTF-8 fails, assume it's a PDF file and try to extract text
            try:
                job_desc = extract_text_from_pdf(job_desc_text)
            except Exception as e:
                st.warning(f"Unable to extract text from Job Description PDF: {str(e)}")
                return
    else:
        st.warning("Please upload a Job Description.")
        return

    # Check if resume_text is uploaded
    if resume_text is not None:
        try:
            # Attempt to decode the resume content as UTF-8
            resume = resume_text.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            # If decoding as UTF-8 fails, assume it's a PDF file and try to extract text
            try:
                resume = extract_text_from_pdf(resume_text)
            except Exception as e:
                st.warning(f"Unable to extract text from Resume PDF: {str(e)}")
                return
    else:
        st.warning("Please upload a resume.")
        return

    df = analyze_str(resume, options)
    df_string = df.applymap(lambda x: ', '.join(x) if isinstance(x, list) else x).to_string(index=False)
    st.write("Analyzing with OpenAI..")
    summary_question = f"Job requirements: {{{job_desc}}}" + f"Resume summary: {{{df_string}}}" + "Please return a summary of the candidate's suitability for this position (limited to 200 words);'"
    summary = ask_openAI(summary_question)
    df.loc[len(df)] = ['Summary', summary]
    extra_info = "Scoring criteria: Top 10 domestic universities +3 points, 985 universities +2 points, 211 universities +1 point, leading company experience +2 points, well-known company +1 point, overseas background +3 points, foreign company background +1 point."
    score_question = f"Job requirements: {{{job_desc}}}" + f"Resume summary: {{{df.to_string(index=False)}}}" + "Please return a matching score (0-100) for the candidate for this job, please score accurately to facilitate comparison with other candidates, '" + extra_info
    score = ask_openAI(score_question)
    df.loc[len(df)] = ['Match Score', score]

    return df

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def ask_openAI(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].text.strip()

def analyze_str(resume, options):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(resume)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    df_data = [{'option': option, 'value': []} for option in options]
    st.write("Fetching information")

    # Create a progress bar and an empty element
    progress_bar = st.progress(0)
    option_status = st.empty()

    for i, option in tqdm(enumerate(options), desc="Fetching information", unit="option", ncols=100):
        question = f"What is this candidate's {option}? Please return the answer in a concise manner, no more than 250 words. If not found, return 'Not provided'"
        docs = knowledge_base.similarity_search(question)
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0.3, model_name="text-davinci-003", max_tokens="2000")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question )
        df_data[i]['value'] = response
        option_status.text(f"Looking for information: {option}")

        # Update the progress bar
        progress = (i + 1) / len(options)
        progress_bar.progress(progress)

    df = pd.DataFrame(df_data)
    st.success("Resume elements retrieved")
    return df

# Set the page title
st.title("🚀 BrightPath: GPT Recruitment Analysis")
st.subheader("With the Help of AI")

# Set default job description and resume information
default_jd = "Business Data Analyst JD: Duties: ..."
default_resume = "Resume: Personal Information: ..."

# Upload job description
jd_text = st.file_uploader("Upload Job Description (Text or PDF)", type=["txt", "pdf"])

# Upload resume
resume_text = st.file_uploader("Upload Candidate Resume (Text or PDF)", type=["txt", "pdf"])

# Parameter input
options = ["Name", "Contact Number", "Gender", "Age", "Years of Work Experience (Number)", "Highest Education", "Undergraduate School Name", "Master's School Name", "Employment Status", "Current Position", "List of Past Employers", "Technical Skills", "Experience Level", "Management Skills"]
selected_options = st.multiselect("Please select options", options, default=options)

# Initialize df
df = None

def get_binary_file_downloader_html(bin_str, file_label='File', file_name='file.txt'):
    """Generate a download link for a binary file."""
    bin_str = bin_str.encode()
    b64 = base64.b64encode(bin_str).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">{file_label}</a>'
    return href

# Analyze button
if st.button("Start Analysis"):
    df = analyze_resume(jd_text, resume_text, selected_options)
    st.subheader("Overall Match Score: "+ df.loc[df['option'] == 'Match Score', 'value'].values[0])
    st.subheader("Detailed Display:")
    st.table(df)

    # Download results as TXT
    txt_result = df.to_csv(sep='\t', index=False, header=None)
    st.markdown(get_binary_file_downloader_html(txt_result, file_label="Download Results", file_name="Recruitment_Results.txt"), unsafe_allow_html=True)

