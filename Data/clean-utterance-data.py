import pandas as pd

# this file is not shared publicly
import_file = "path/to/all_combined_total_features_renamed_spring_only.csv"

data = pd.read_csv(import_file)

columns_to_keep = [
        # metadata
        'ObsID',
        'UtteranceID',
        'TeacherID',
        # text of the utterance
        'TranscribedUtterance',
        # labels for talk features
        'IsQuestion',
        'IsInstructionalUtterance',
        'IsInstructionalStatement',
        'IsDisciplinaryUtterance',
        'IsDisciplinaryStatement',
        'IsEvaluationFollowupIncluded',
        'IsEvaluationElaborated',
        'IsEvaluationValencePositive',
        'CombinedAuthCogUptake',
        'CogLevel',
        'Uptake',
        'IsGoalSpecified',
        'IsDisciplinaryTermsPresent',
        'IsInstructionalQuestion',
        'IsDisciplinaryQuestion',
        'IsStudentResponsePresent',
        'Authenticity',
        'IsSerialQuestion',
        ]

data_export = data[columns_to_keep]

# this file is not shared publicly
data_export.to_csv('utterance-data-with-labels.csv', index = False)
