# python-project-seven-machinelearning

## project overview
    In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to 
    widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential 
    information entered into the public record, including tens of thousands of emails and detailed financial data for 
    top executives. In this project, I will play detective, and put my new skills(machine learning) to use by building 
    a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

## features of the dataset
        financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',        'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

        email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

        POI label: [‘poi’] (boolean, represented as integer)

## tool
    main tool: python 2
    libraries： scikit-learn

### to see my code, please cilck "poi_id.py".
### to read my project report, please click "project-seven-machine-learning.pdf".
