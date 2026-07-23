import pandas as pd
import random

random.seed(42)

billing_subjects = [
    "Payment Failed", "Refund Pending", "Invoice Missing",
    "Subscription Charged", "Incorrect Bill",
    "Duplicate Payment", "Refund Request", "Billing Issue"
]

technical_subjects = [
    "Unable to Login", "Application Crash", "Server Down",
    "Database Error", "Website Not Loading",
    "Password Reset", "API Error", "Network Issue"
]

hr_subjects = [
    "Leave Request", "Payslip Required", "Salary Issue",
    "Offer Letter", "Attendance Correction",
    "Holiday List", "Resignation", "Joining Date"
]

general_subjects = [
    "Office Timings", "Contact Information",
    "Business Hours", "Company Address",
    "General Enquiry", "Need Help",
    "Product Information", "Feedback"
]

billing_templates = [
    "Hello Team, my payment of ₹{amount} was deducted but the order is still pending.",
    "I cancelled my subscription but my refund has not been received.",
    "Please share the invoice for Order #{order}.",
    "I was charged twice for the same transaction.",
    "The billing amount seems incorrect. Kindly verify."
]

technical_templates = [
    "The application crashes whenever I try to login.",
    "The website is not loading since morning.",
    "Server is down and users cannot access the application.",
    "Password reset link is not working.",
    "I am getting a database connection error."
]

hr_templates = [
    "Please approve my leave request for next week.",
    "Kindly send my payslip for this month.",
    "I have not received my offer letter yet.",
    "Attendance has been marked incorrectly.",
    "Need clarification regarding my salary."
]

general_templates = [
    "What are your office working hours?",
    "Can you share your company address?",
    "How can I contact customer support?",
    "Need more information about your services.",
    "Please provide your contact details."
]

sources = ["Email", "Portal", "Live Chat"]

priorities = ["Normal", "High"]

rows = []
ticket_id = 1001


def create_ticket(category, subjects, templates):
    global ticket_id

    subject = random.choice(subjects)

    body = random.choice(templates).format(
        amount=random.randint(500, 15000),
        order=random.randint(10000, 99999)
    )

    rows.append([
        ticket_id,
        subject,
        body,
        category,
        random.choice(priorities),
        random.choice(sources)
    ])

    ticket_id += 1


# 125 tickets for each category
for _ in range(125):
    create_ticket("Billing", billing_subjects, billing_templates)

for _ in range(125):
    create_ticket("Technical", technical_subjects, technical_templates)

for _ in range(125):
    create_ticket("HR", hr_subjects, hr_templates)

for _ in range(125):
    create_ticket("General", general_subjects, general_templates)

df = pd.DataFrame(rows, columns=[
    "ticket_id",
    "subject",
    "body",
    "category",
    "priority",
    "source"
])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save CSV
df.to_csv("data/support_tickets.csv", index=False)

print("=" * 50)
print("Dataset Generated Successfully!")
print(f"Total Tickets : {len(df)}")
print("=" * 50)
print(df.head())