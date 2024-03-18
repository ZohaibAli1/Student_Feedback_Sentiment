import csv
from flask import Flask, render_template
csv_filename_subjects = 'subjects.csv'

def add_subject_to_csv(subject_name):
    with open(csv_filename_subjects, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([subject_name])
    return render_template('success-page.html')

def update_subject_in_csv(subject_name, new_subject_name):
    subject_records = []

    with open(csv_filename_subjects, mode='r') as file:
        reader = csv.reader(file)
        subject_records = list(reader)

    updated = False
    for record in subject_records:
        if record[0] == subject_name:
            record[0] = new_subject_name
            updated = True

    if updated:
        with open(csv_filename_subjects, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(subject_records)
        return render_template('update.html')
    else:
        return "Subject not found"

def delete_subject_from_csv(subject_name):
    subject_records = []

    with open(csv_filename_subjects, mode='r') as file:
        reader = csv.reader(file)
        subject_records = list(reader)

    updated_records = [record for record in subject_records if record[0] != subject_name]

    with open(csv_filename_subjects, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_records)
    return render_template('delete_html.html')
