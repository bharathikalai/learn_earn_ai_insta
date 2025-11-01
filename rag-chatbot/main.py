from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# Define file path for the PDF
file_path = "Tamil_Nadu_Shop_Invoice_Expanded.pdf"

# Create PDF document
doc = SimpleDocTemplate(file_path, pagesize=A4)
styles = getSampleStyleSheet()
elements = []

# Title and Date
title = Paragraph('<font color="darkblue"><b>Tamil Nadu Shop Invoice</b></font>', styles["Title"])
date = Paragraph('<b>Date:</b> February 2026', styles["Normal"])
elements.extend([title, Spacer(1, 10), date, Spacer(1, 12)])

# Table data with expanded columns
data = [
    ["Customer Name", "Item", "Quantity", "Unit Price", "Total", "Payment Method", "Invoice ID", "Date"],
    ["Vijay TVK", "TVK Membership", "1", "₹500", "₹500", "Cash", "INV001", "Feb 2026"],
    ["Stalin DMK", "DMK Donation", "1", "₹200", "₹200", "Card", "INV002", "Feb 2026"],
    ["Seeman NTK", "NTK Sticker", "1", "₹1", "₹1", "UPI", "INV003", "Feb 2026"],
    ["EPS ADMK", "ADMK Flag", "1", "₹201", "₹201", "Cash", "INV004", "Feb 2026"],
    ["Annamalai BJP", "BJP Badge", "1", "₹100", "₹100", "Card", "INV005", "Feb 2026"],
    ["", "", "", "", "", "", "", ""],
    ["<b>Overall Total</b>", "", "", "", "<b>₹1002</b>", "", "", ""]
]

# Create table with colors
table = Table(data, colWidths=[100, 100, 50, 60, 60, 70, 70, 70])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FFD700")),  # Header
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTNAME', (0, 6), (-1, 6), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#FFDDC1")),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#FFABAB")),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#FFC3A0")),
    ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor("#D5AAFF")),
    ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#85E3FF")),
    ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor("#90EE90")),  # Total row
]))

elements.append(table)
elements.append(Spacer(1, 20))

# Footer
footer = Paragraph('<font color="grey">Thank you for shopping with Tamil Nadu Shop!</font>', styles["Normal"])
elements.append(footer)

# Build PDF
doc.build(elements)

file_path
