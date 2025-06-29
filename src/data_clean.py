import pandas as pd

data = pd.read_csv("../data/raw/data.csv", sep=';', engine='python', encoding='utf-8')

data.columns = data.columns.str.strip()

data.rename(columns=lambda x: x.strip(), inplace=True)

# data['Target'] = data['Target'].apply(lambda x: 0 if x == 'Dropout' else 1)

# Map to numbers
data['Target'] = data['Target'].map({
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
})

drop_cols = [
    'Unemployment rate', 'Inflation rate', 'GDP',
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (without evaluations)',
    "Mother's occupation", "Father's occupation",
    'Nacionality',
    'Application mode', 'Course'
]
clean_data = data.drop(columns=drop_cols)

clean_data.dropna(inplace=True)

clean_data['Curricular units 1st sem (pass rate)'] = (clean_data['Curricular units 1st sem (approved)'] / clean_data['Curricular units 1st sem (enrolled)']).fillna(0)

clean_data['Curricular units 2nd sem (pass rate)'] = (clean_data['Curricular units 2nd sem (approved)'] / clean_data['Curricular units 2nd sem (enrolled)']).fillna(0)

cols = clean_data.columns.tolist()

cols.remove('Curricular units 1st sem (pass rate)')
cols.remove('Curricular units 2nd sem (pass rate)')

cols.insert(cols.index('Curricular units 1st sem (approved)') + 1, 'Curricular units 1st sem (pass rate)')
cols.insert(cols.index('Curricular units 2nd sem (approved)') + 1, 'Curricular units 2nd sem (pass rate)')

clean_data = clean_data[cols]

clean_data.to_csv('../data/processed/clean_data.csv', index=False)