# BrightPath: GPT Recruitment Analysis

This Streamlit application analyzes resumes using GPT-3 for matching with job descriptions. It allows users to upload a job description and a candidate's resume (in text or PDF format) and select various resume elements for analysis. The application uses OpenAI's GPT-3 model to extract relevant information from the resume based on the selected elements and the provided job description.

## Features

- **Resume Analysis**: Analyze candidate resumes for various elements such as name, contact information, education, skills, experience, etc.
- **Job Description Matching**: Generate a match score indicating how well a candidate's resume matches the provided job description.
- **Download Results**: Download the analysis results in text format for further review.

## Prerequisites

Before running the application, make sure you have the following:

- Python installed on your machine
- OpenAI API key
- Streamlit library
- pandas, tqdm, langchain Python libraries

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/rizwaniscoder/brightpath-gpt-recruitment.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd brightpath-gpt-recruitment
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the project directory and add your OpenAI API key:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

1. **Run the application:**

    ```bash
    streamlit run main.py
    ```

2. **Access the application:**

    Open your web browser and go to `http://localhost:8501`.

3. **Upload Job Description:**

    - Upload a job description in text or PDF format.

4. **Upload Candidate Resume:**

    - Upload a candidate's resume in text or PDF format.

5. **Select Resume Elements:**

    - Choose the resume elements you want to analyze from the provided list.

6. **Start Analysis:**

    - Click on the "Start Analysis" button to begin the analysis process.

7. **View Results:**

    - Once the analysis is complete, view the overall match score and detailed analysis results.
    - You can also download the analysis results in text format for further review.

## License

This project is licensed under the [MIT License](LICENSE).
