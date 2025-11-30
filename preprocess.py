# preprocess.py
import pandas as pd

def safe_divideass(mark, total):
    mark = 0 if pd.isna(mark) else float(mark)
    return (mark / total)*3.75 if total != 0 else 0

def safe_dividequi(mark, total):
    mark = 0 if pd.isna(mark) else float(mark)
    return (mark / total)*2 if total != 0 else 0

def preprocess_dataset(file_path="marks_dataset.xlsx"):
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    all_sheets_scores = []

    for sheet in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)

        # Identify Assignment and Quiz columns
        assignment_cols = [c for c in df.columns if c.startswith('As:')]
        quiz_cols = [c for c in df.columns if c.startswith('Qz:')]

        # Get total row
        total_row = df[df.iloc[:,0] == 'Total'].iloc[0]

        assignment_totals = [float(total_row[col]) if pd.notna(total_row[col]) else 0 for col in assignment_cols]
        quiz_totals = [float(total_row[col]) if pd.notna(total_row[col]) else 0 for col in quiz_cols]

        student_rows = df.iloc[3:].copy()

        # Assignments
        assignments_all = []
        for idx, row in student_rows.iterrows():
            percentages = [safe_divideass(row[col], assignment_totals[i]) for i, col in enumerate(assignment_cols)]
            top4 = sorted(percentages, reverse=True)[:4]
            assignments_all.append(sum(top4))

        # Quizzes
        quizzes_all = []
        for idx, row in student_rows.iterrows():
            percentages = [safe_dividequi(row[col], quiz_totals[i]) for i, col in enumerate(quiz_cols)]
            top5 = sorted(percentages, reverse=True)[:5]
            quizzes_all.append(sum(top5))

        student_rows['Assignments'] = assignments_all
        student_rows['Quizzes'] = quizzes_all
        student_rows['Mid1'] = student_rows['S-I'].fillna(0).astype(float)
        student_rows['Mid2'] = student_rows['S-II'].fillna(0).astype(float)
        student_rows['Final'] = student_rows['Final'].fillna(0).astype(float)
        student_rows['Total'] = student_rows['Assignments'] + student_rows['Quizzes'] + student_rows['Mid1'] + student_rows['Mid2'] + student_rows['Final']

        all_sheets_scores.append(student_rows[['Assignments','Quizzes','Mid1','Mid2','Final','Total']])

    combined_final_scores = pd.concat(all_sheets_scores, ignore_index=True)
    combined_final_scores.fillna(0, inplace=True)
    combined_final_scores.to_csv("preprocessed_dataset.csv", index=False)
    
    return combined_final_scores
