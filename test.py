from flask import Flask, jsonify, render_template
import csv
from collections import defaultdict


def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def load_assignment_records(file_path):
    records = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row['Course ID']] = {
                'course_name': row['course_name'],
                'instructor': row['username']
            }
    return records


def calculate_scores(data):
    course_scores = defaultdict(lambda: {'positive': 0, 'negative': 0, 'total': 0})

    for row in data:
        course_id = row['Course ID']
        course_scores[course_id]['total'] += 1

        positive_fields = ['Learning Outcomes', 'Workload Manageable', 'Well Organized', 'Student Participation',
                           'Student Progress', 'Well Structured', 'Encouraged Participation', 'Conducive to Learning',
                           'Learning Materials', 'Recommended Reading', 'Library Resources', 'Online Resources',
                           'Interest Stimulated', 'Course Pace', 'Clear Presentation', 'Assessment Reasonable',
                           'Feedback Timely', 'Feedback Helpful']

        for field in positive_fields:
            if row[field] in ['1', '2']:
                course_scores[course_id]['positive'] += 1
            elif row[field] in ['-1', '-2']:
                course_scores[course_id]['negative'] += 1

        if 'bad' in row['BestFeatures'].lower():
            course_scores[course_id]['negative'] += 1
        if 'good' in row['ImprovementAreas'].lower():
            course_scores[course_id]['negative'] += 1

    return course_scores


def analyze_courses(scores, assignment_records):
    best_course = max(scores, key=lambda x: scores[x]['positive'] / scores[x]['total'])
    most_annoying = max(scores, key=lambda x: scores[x]['negative'] / scores[x]['total'])
    most_difficult = max(scores, key=lambda x: (scores[x]['negative'] - scores[x]['positive']) / scores[x]['total'])

    results = {
        'best': {
            'id': best_course,
            'name': assignment_records[best_course]['course_name'],
            'instructor': assignment_records[best_course]['instructor']
        },
        'most_annoying': {
            'id': most_annoying,
            'name': assignment_records[most_annoying]['course_name'],
            'instructor': assignment_records[most_annoying]['instructor']
        },
        'most_difficult': {
            'id': most_difficult,
            'name': assignment_records[most_difficult]['course_name'],
            'instructor': assignment_records[most_difficult]['instructor']
        }
    }

    return results


