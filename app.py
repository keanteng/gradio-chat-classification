import gradio as gr
from src.model import classify_loan
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

with gr.Blocks(
    title="Loan Approval Classification",
    theme=gr.themes.Base(),
    css=".gradio-container { background-color: #f0f0f0; }"
) as demo:
    gr.Markdown("# Loan Approval Classification")
    gr.Markdown("This application classifies loan applications based on user input. Powered by Gemini 2.5 Flash.")

    with gr.Row():
        with gr.Column():
            person_age = gr.Number(value = 22, label="Age")
            person_gender = gr.Radio(['male', 'female'], value = 'female', label="Gender")
            person_education=gr.Radio(['high school', 'bachelor', 'master', 'associate', 'doctorate'], value = 'master', label="Education")
            person_income = gr.Number(label="Income", value = 71498)
            person_emp_exp = gr.Number(label="Employment Experience", value = 1)
            person_home_ownership = gr.Radio(['own', 'rent', 'mortgage', 'other'], value = 'rent', label="Home Ownership")
            loan_amnt = gr.Number(label="Loan Amount", value = 35000)
            loan_intent = gr.Radio(['personal', 'home_improvement', 'debt consolidation', 'education', 'medical', 'venture'], value = 'personal', label="Loan Intent")
            loan_int_rate = gr.Number(value = 16.02, label="Loan Interest Rate")
            loan_percent_income = gr.Number(value = 0.49, label="Loan Percentage of Income")
            cb_cred_hist_length = gr.Number(value = 3, label="Credit History Length")
            credit_score = gr.Number(value = 561, label="Credit Score")
            previous_loan_defaults_on_file = gr.Radio(['yes', 'no'], value = 'no', label="Previous Loan Defaults on File")
            submit_button = gr.Button("Submit Form")
        with gr.Column():
            classification_output = gr.TextArea(label="Output - Gemini 2.5 Flash", interactive=False)

    submit_button.click(
        classify_loan, 
        inputs=[
            person_age,
            person_gender,
            person_education,
            person_income,
            person_emp_exp,
            person_home_ownership,
            loan_amnt,
            loan_intent,
            loan_int_rate,
            loan_percent_income,
            cb_cred_hist_length,
            credit_score,
            previous_loan_defaults_on_file
        ], 
        outputs=classification_output
    )

if __name__ == "__main__":
    demo.launch()