import gradio as gr
from src.model import classify_loan

with gr.Blocks(
    title="Loan Approval Classification",
    theme=gr.themes.Base(),
    css=".gradio-container { background-color: #f0f0f0; }"
) as demo:
    gr.Markdown("# Loan Approval Classification")
    gr.Markdown("This application classifies loan applications based on user input. Powered by Gemini 2.5 Flash.")

    with gr.Row():
        with gr.Column():
            person_age = gr.Number(label="Age")
            person_gender = gr.Radio(['male', 'female'], label="Gender")
            person_education=gr.Radio(['high school', 'bachelor', 'master', 'associate', 'doctorate'], label="Education")
            person_income = gr.Number(label="Income")
            person_emp_exp = gr.Number(label="Employment Experience")
            person_home_ownership = gr.Radio(['own', 'rent', 'mortgage', 'other'], label="Home Ownership")
            loan_amnt = gr.Number(label="Loan Amount")
            loan_intent = gr.Radio(['personal', 'home_improvement', 'debt consolidation', 'education', 'medical', 'venture'], label="Loan Intent")
            loan_int_rate = gr.Number(label="Loan Interest Rate")
            loan_percent_income = gr.Number(label="Loan Percentage of Income")
            cb_cred_hist_length = gr.Number(label="Credit History Length")
            credit_score = gr.Number(label="Credit Score")
            previous_loan_defaults_on_file = gr.Radio(['yes', 'no'], label="Previous Loan Defaults on File")
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