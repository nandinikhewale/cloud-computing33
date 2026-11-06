"""
generate_sample_pdfs.py - Generates 5 realistic college PDF documents in data/
Includes Academic Regulations, Examination Guidelines, Student Handbook,
Placement Guidelines, and Department Rules.
"""

import os
from fpdf import FPDF


class CollegePDF(FPDF):
    def __init__(self, title_text: str):
        super().__init__()
        self.doc_title = title_text

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "XYZ INSTITUTE OF TECHNOLOGY & SCIENCE", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 10)
        self.cell(0, 6, "Autonomous Institution | Approved by AICTE & Affiliated to State University", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 51, 102)
        self.cell(0, 8, self.doc_title, border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.line(10, 36, 200, 36)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} - Official College Publication", align="C")

    def add_section(self, heading: str, body: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(20, 40, 80)
        self.cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(30, 30, 30)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, body)
        self.ln(3)


def create_all_college_pdfs(output_dir: str = "./data"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Academic Regulations
    pdf1 = CollegePDF("ACADEMIC REGULATIONS & CURRICULUM FRAMEWORK")
    pdf1.add_page()
    pdf1.add_section(
        "1. Minimum Attendance Requirements",
        "A student must maintain a minimum attendance of 75% in aggregate across all registered courses in a semester to be eligible to appear for the semester end examinations.\n"
        "Students having attendance between 65% and 74% may apply for attendance condonation solely on genuine medical grounds or sanctioned institutional representation. "
        "A non-refundable condonation fee of Rs. 1,000 must be remitted along with a valid medical certificate issued by a registered medical practitioner.\n"
        "Students with attendance below 65% are strictly not eligible for condonation and shall be detained, requiring them to repeat the entire semester in the subsequent academic year."
    )
    pdf1.add_section(
        "2. Degree Credit Requirements",
        "For the award of the Bachelor of Technology (B.Tech) degree, a candidate must successfully complete and earn a minimum total of 160 credits over 8 semesters.\n"
        "Lateral entry students admitted directly into the second year (third semester) must earn a minimum of 120 credits across 6 semesters.\n"
        "The curriculum distribution includes: Basic Sciences (24 credits), Engineering Sciences (20 credits), Professional Core (60 credits), Professional Electives (18 credits), Open Electives (12 credits), Humanities & Management (12 credits), and Mandatory Projects/Internships (14 credits)."
    )
    pdf1.add_section(
        "3. Grading System and Passing Standards",
        "Academic performance is assessed using a 10-point letter grading scale:\n"
        "  - O (Outstanding): Grade Point 10 (Marks >= 90%)\n"
        "  - A+ (Excellent): Grade Point 9 (Marks 80% to 89%)\n"
        "  - A (Very Good): Grade Point 8 (Marks 70% to 79%)\n"
        "  - B+ (Good): Grade Point 7 (Marks 60% to 69%)\n"
        "  - B (Above Average): Grade Point 6 (Marks 50% to 59%)\n"
        "  - C (Passing): Grade Point 5 (Marks 40% to 49%)\n"
        "  - F (Fail / Backlog): Grade Point 0 (Marks < 40%)\n"
        "A student must secure a minimum of 40% marks in Semester End Examination and 40% overall aggregate to earn passing credit in a course."
    )
    pdf1.add_section(
        "4. Semester Registration Procedure",
        "Every student is required to complete formal semester course registration within 7 calendar days from the official date of semester commencement.\n"
        "Late registration is permitted up to 14 days with an administrative late fee of Rs. 500. Registration beyond 14 days is not entertained without Dean of Academics approval."
    )
    pdf1.output(os.path.join(output_dir, "academic_regulations.pdf"))

    # 2. Examination Guidelines
    pdf2 = CollegePDF("EXAMINATION GUIDELINES & EVALUATION POLICIES")
    pdf2.add_page()
    pdf2.add_section(
        "1. Examination Revaluation and Recounting Procedure",
        "Students who are not satisfied with their semester end examination grades may apply for paper revaluation or recounting within 15 calendar days from the date of declaration of results on the college portal.\n"
        "The revaluation fee is Rs. 500 per theoretical course, and recounting fee is Rs. 200 per course.\n"
        "If the difference between the original marks and the revaluated marks is 5% or more, the higher mark will be adopted as the revised final grade.\n"
        "Revaluation is strictly not applicable for practical laboratory examinations, seminar evaluations, and project viva-voce examinations."
    )
    pdf2.add_section(
        "2. Backlog Examinations & Supplementary Schedule",
        "Supplementary examinations for clearing backlog courses are conducted twice each academic year, concurrently with the regular Odd and Even semester end examinations.\n"
        "A student with backlogs can register for a maximum of 4 backlog theory subjects per examination session alongside their regular current-semester subjects.\n"
        "Students must submit the supplementary exam registration form and pay a backlog examination fee of Rs. 350 per subject before the notified deadline."
    )
    pdf2.add_section(
        "3. Hall Ticket Eligibility & Issuance",
        "Hall tickets for end-semester examinations will be issued 3 days prior to the examination start date via the student portal.\n"
        "Eligibility criteria for hall ticket release:\n"
        "  a) Verified minimum 75% attendance (or approved condonation).\n"
        "  b) Full settlement of tuition fees and library dues.\n"
        "  c) No pending disciplinary suspension orders from the Proctorial Board."
    )
    pdf2.add_section(
        "4. Malpractice Penalties and Proctorial Action",
        "Possession of mobile phones, smartwatches, programmable calculators, cheat sheets, or unauthorized electronic gadgets inside the exam hall constitutes serious malpractice.\n"
        "Category 1 Malpractice (Possession without use): Cancellation of that specific paper.\n"
        "Category 2 Malpractice (Copying / Use of external aid): Cancellation of all written exams of that session plus debarment from examinations for 1 full academic year."
    )
    pdf2.output(os.path.join(output_dir, "examination_guidelines.pdf"))

    # 3. Student Handbook
    pdf3 = CollegePDF("STUDENT CODE OF CONDUCT & CAMPUS LIFE HANDBOOK")
    pdf3.add_page()
    pdf3.add_section(
        "1. Campus Timings & General Working Hours",
        "College instructional hours are from 9:00 AM to 4:30 PM, Monday through Friday.\n"
        "Administrative and student welfare counters operate between 9:30 AM and 4:00 PM on all instructional days.\n"
        "The campus gates are closed for entry at 9:15 AM to ensure punctuality."
    )
    pdf3.add_section(
        "2. Central Library Rules & Lending Policies",
        "The Central Library remains open from 8:00 AM to 8:00 PM on all working days and 9:00 AM to 2:00 PM on Saturdays.\n"
        "Undergraduate students are eligible to borrow up to 4 books concurrently for a loan period of 14 days.\n"
        "An overdue overdue fine of Rs. 2 per book per day will be levied on delayed returns.\n"
        "Reference books, journals, and thesis dissertations are restricted to reading room consultation and cannot be issued out."
    )
    pdf3.add_section(
        "3. Anti-Ragging Policy & Institutional Zero Tolerance",
        "Ragging in any form (physical, psychological, cyber, or verbal) is strictly prohibited both inside and outside the college campus.\n"
        "The college maintains a Zero Tolerance Policy under UGC regulations.\n"
        "National Anti-Ragging 24x7 Toll-Free Helpline: 1800-180-5522.\n"
        "College Anti-Ragging Squad Hotline: +91-9876543210 (ext: 204).\n"
        "Any student found guilty of ragging shall face immediate suspension, FIR filing with local police, and expulsion."
    )
    pdf3.add_section(
        "4. Student Clubs, Sports & Extracurriculars",
        "Every student is encouraged to participate in at least one registered student body:\n"
        "  - Technical: Google Developer Student Club (GDSC), Robotics & AI Club, Open Source Guild.\n"
        "  - Cultural & Literary: Rotaract Youth, Music Society, Theatre & Dramatic Guild, Debating Society.\n"
        "  - Sports: Cricket, Football, Basketball, Badminton, and Athletics teams with dedicated NIS-certified coaches."
    )
    pdf3.add_section(
        "5. Dress Code and ID Card Compliance",
        "Students are expected to dress in modest, professional attire suited to an academic learning environment.\n"
        "Identity cards (ID cards) issued by the college must be worn visibly around the neck using the official lanyard at all times while present on campus."
    )
    pdf3.output(os.path.join(output_dir, "student_handbook.pdf"))

    # 4. Placement Guidelines
    pdf4 = CollegePDF("TRAINING & PLACEMENT CELL POLICY GUIDELINES")
    pdf4.add_page()
    pdf4.add_section(
        "1. Placement Eligibility Criteria",
        "To be eligible for participation in campus placement recruitment drives, students must fulfill:\n"
        "  a) Minimum cumulative grade point average (CGPA) of 6.50 through the 6th semester.\n"
        "  b) No active backlogs (standing arrears) at the time of appearing for the company test/interview.\n"
        "  c) Minimum 85% attendance record in pre-placement soft-skills and aptitude training programs."
    )
    pdf4.add_section(
        "2. Offer Tier System (Dream & Super-Dream Offers)",
        "The Training & Placement Cell follows a structured tier policy:\n"
        "  - Regular Offer: Compensation Packages up to Rs. 6.0 LPA.\n"
        "  - Dream Offer: Packages between Rs. 6.01 LPA and Rs. 11.99 LPA.\n"
        "  - Super-Dream Offer: Packages of Rs. 12.0 LPA and above.\n"
        "Policy: A student securing a Regular Offer remains eligible to apply exclusively for Dream or Super-Dream companies. "
        "Once a student secures a Dream Offer, they are solely eligible for Super-Dream opportunities. Securing a Super-Dream offer concludes their placement drives."
    )
    pdf4.add_section(
        "3. Mandatory Registration Documents",
        "Students appearing for campus drives must bring 2 copies of verified resumes, original 10th and 12th standard mark sheets, all semester grade sheets, 4 recent passport-size photographs, government ID (Aadhaar/PAN/Passport), and official college ID card."
    )
    pdf4.add_section(
        "4. Discipline & Code of Conduct during Drives",
        "Full formal business attire is mandatory for all interviews.\n"
        "Registering for a company drive and failing to appear without prior written Dean permission will result in blacklisting from 3 subsequent company recruitment drives."
    )
    pdf4.output(os.path.join(output_dir, "placement_guidelines.pdf"))

    # 5. Department Rules
    pdf5 = CollegePDF("DEPARTMENTAL ACADEMIC & LABORATORY REGULATIONS")
    pdf5.add_page()
    pdf5.add_section(
        "1. Laboratory Safety Rules & Operational Standards",
        "Safety is paramount across all department laboratories:\n"
        "  - Students must wear white lab coats and closed-toe leather/safety shoes during chemistry, physics, and mechanical workshops.\n"
        "  - Electrostatic Discharge (ESD) wrist straps are mandatory when handling microcontrollers and circuit hardware in electronics laboratories.\n"
        "  - Food, beverages, and mobile phones are strictly forbidden inside computing and hardware laboratories."
    )
    pdf5.add_section(
        "2. Internal Assessment Weightage Breakdown",
        "Each course evaluation carries a total of 100 marks divided as follows:\n"
        "  - Continuous Internal Assessment (CIA): 40 Marks\n"
        "      * Mid-Term Test 1 & 2: 25 Marks\n"
        "      * Continuous Quizzes and Technical Assignments: 10 Marks\n"
        "      * Classroom Attendance & Class Participation: 5 Marks\n"
        "  - Semester End Examination (SEE): 60 Marks"
    )
    pdf5.add_section(
        "3. Mini-Project and Major Project Submission Rules",
        "Final-year mini-project reports must adhere strictly to IEEE two-column documentation standards.\n"
        "Soft copy and two hard-bound copies signed by the project guide and Head of Department (HOD) must be submitted at least 14 days prior to the commencement of viva-voce examinations.\n"
        "Plagiarism in project code or report text exceeding 15% as checked via Turnitin will lead to automatic rejection of the project."
    )
    pdf5.output(os.path.join(output_dir, "department_rules.pdf"))

    print(f"Successfully generated 5 official college PDF documents in '{output_dir}'.")


if __name__ == "__main__":
    create_all_college_pdfs("./data")
