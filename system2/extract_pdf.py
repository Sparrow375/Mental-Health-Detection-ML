import pypdf
reader = pypdf.PdfReader('C:/Users/embar/OneDrive-N/D0cuments/GitHub/Mental-Health-Detection-ML/StudentLife_Validation_Report.pdf')
text = ""
for page in reader.pages:
    text += page.extract_text()
with open('tmp_report_format.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print("Extracted PDF to tmp_report_format.txt")
