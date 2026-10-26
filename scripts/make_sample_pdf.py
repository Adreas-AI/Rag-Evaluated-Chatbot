"""Generate sample/employee_handbook.pdf: a FICTIONAL 3-page document for demos.

Each page covers a different topic, so answers can be checked against page citations.
Run: python scripts/make_sample_pdf.py   (needs reportlab, see requirements-dev.txt)
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "sample" / "employee_handbook.pdf"

PAGES = [
    (
        "Northwind Coffee Co. - Employee Handbook (fictional sample)",
        [
            "This handbook is a fictional example created to demonstrate the RAG chatbot.",
            "Working hours. The standard working week is 37.5 hours, Monday to Friday. "
            "Core hours, when everyone should be available, are 10:00 to 15:00.",
            "Remote work. Employees may work remotely up to 3 days per week after "
            "completing their 3-month probation period. Remote days must be agreed with "
            "the team lead at least one week in advance.",
            "Office. The Athens office is open from 08:00 to 19:00. Visitors must be "
            "registered at reception.",
        ],
    ),
    (
        "Leave and time off",
        [
            "Annual leave. Full-time employees receive 25 days of paid annual leave per "
            "calendar year. Up to 5 unused days can be carried over to the next year and "
            "must be used by 31 March.",
            "Sick leave. Employees must notify their team lead before 10:00 on the first "
            "day of absence. A doctor's note is required for absences longer than 3 days.",
            "Parental leave. Parents are entitled to 16 weeks of paid parental leave, "
            "which can be taken within the first 12 months after birth or adoption.",
        ],
    ),
    (
        "Expenses and security",
        [
            "Expenses. Business expenses up to 200 euros can be approved by the team lead. "
            "Higher amounts need approval from the finance department. Receipts must be "
            "submitted within 30 days through the expenses portal.",
            "Equipment. Every employee receives a laptop and a headset. Lost or stolen "
            "equipment must be reported to IT within 24 hours.",
            "Security. Passwords must be at least 14 characters long and two-factor "
            "authentication is mandatory for all company accounts. Never share passwords "
            "or API keys by email or chat.",
        ],
    ),
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Title"] if i == 0 else styles["Heading1"]))
        story.append(Spacer(1, 12))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(OUT), pagesize=A4, title="Employee Handbook (fictional sample)",
                      author="Rag-Evaluated-Chatbot").build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
