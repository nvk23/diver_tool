import os
import sys
import subprocess
import numpy as np
import pandas as pd
print(pd.__version__)
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import datetime
import csv
from PIL import Image
from io import StringIO
from google.cloud import storage

# from api import api_call  # will not work unless on VPN

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'card-diver-1ed90ae12ffe.json'

# Methods to pull from Google Cloud storage - put in utils file eventually
def blob_as_csv(bucket, path, sep='\s+', header='infer'):  # reads in file from google cloud folder
    blob = bucket.get_blob(path)
    blob = blob.download_as_bytes()
    blob = str(blob, 'utf-8')
    blob = StringIO(blob)
    df = pd.read_csv(blob, sep=sep, header=header)
    return df

def blob_as_xml(bucket, path, xpath):  # reads in file from google cloud folder
    blob = bucket.get_blob(path)
    blob = blob.download_as_bytes()
    blob = str(blob, 'utf-8')
    blob = StringIO(blob)
    df = pd.read_xml(blob, xpath = xpath)
    return df

def get_gcloud_bucket(bucket_name):  # gets folders from Google Cloud
    storage_client = storage.Client(project='card-diver')
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

# Sets the value in multiselect session state back to an empty list
def clear_multi():
    st.session_state.multiselect1 = []
    st.session_state.multiselect2 = []

# Maps rows to dictionary values and combines values with same labels 
def replace_vals(col_name, redcap_df):
    # dictionary values
    subject_vals = diver_dict.loc[col_name].to_list()
    subject_vals = subject_vals[0].split('|')

    subject_dict = {}
    for item in subject_vals:
        item_list = item.split(',')
        subject_dict[item_list[0].strip()] = item_list[1].strip()

    # replace labels with full name
    for labels in redcap_df['ItemOID']:
        if col_name in labels:
            if labels.count("_") > 1:
                split_labels = labels.split('_')
                subject_number = split_labels[-1]
                subject_index = redcap_df['ItemOID'].loc[lambda x: x==labels].index.values

                for i in subject_index:
                    if redcap_df['Value'].iloc[i] == '1':
                        redcap_df['Value'].iloc[i] = subject_dict[subject_number]

                # combine rows with similar col_names here
                redcap_df.replace(labels, col_name.upper(), inplace = True)

# Displays alphabetized and cleaned table of study or project
def display_clean(study, project_name = False, which_datatype = False, path_display = False):
    zeroVals = np.where(study["Value"] == '0')

    remove_cols = ['study_information_timestamp', 'project_information_complete', 'study_information_complete',
                    'reference_file_information_complete']

    cleaned_study = study.drop(index = zeroVals[0])
    cleaned_study['Value'].replace('1', 'Yes', inplace = True)

    for item in remove_cols: 
        try: 
            remove_idx = np.where(study["Label"] == item)
            cleaned_study.drop(index = remove_idx[0], inplace = True)
        except:
            pass
    
    if project_name:
        if which_datatype:
            st.markdown(f'**_Project with {which_datatype} datatype: {project_name}_**')
        else:
            st.markdown(f'**_Project {project_name}_**')
    elif not path_display:
        remove_study = ['project_id', 'protocol_path', 'dictionary', 'dictionary_path', 'total_number',
                        'project_contact', 'project_internal_path', 'project_external_path', 'project_external_internal',
                        'project_notes', 'availability_mmse', 'availability_moca', 'availability_neurocog',
                        'availability_med_status', 'clinical_study_status', 'imaging_type','imaging_derived_raw']
        for item in remove_study: 
            remove_idx = np.where(study["Label"] == item)
            cleaned_study.drop(index = remove_idx[0], inplace = True)
        st.markdown(f'#### Study Overview: {cleaned_study.loc[cleaned_study["Label"] == "study_id", "Value"][1]}')
    else:
        st.markdown(f'#### Study Overview: {cleaned_study.loc[cleaned_study["Label"] == "study_id", "Value"][1]}')

    cleaned_study = cleaned_study.groupby('Label',as_index=False).agg({'Value':list})
    cleaned_study['Label'] = cleaned_study['Label'].str.lower()

    st.dataframe(cleaned_study.sort_values('Label').reset_index(drop=True), use_container_width=True)


# Processes individual studies and projects within them
def process_forms(study, disease = False, study_choice = False, datatype = False): 
    satisfy_disease = False
    satisfy_study = False

    # only outputs studies with selected name or target disease
    if disease in study['Value'].values:
        satisfy_disease = True
    if study_choice in study['Value'].values:
        satisfy_study = True
    
    if satisfy_disease or satisfy_study:
        # finds indices of all projects within study
        projects = study['Label'].loc[lambda x: x=='project_id'].index.values
        projects = list(projects)
        projects.append(len(study))

        if not form_display:
            display_clean(study)
        else: # if user selects path labels to display
            result = pd.DataFrame()

            # need study ID to differentiate multiple studies
            if "study_id" not in form_display:
                form_display.append('study_id')

            # only add project ID where necessary based on options selected
            project_choices = ['project_external_path', 'project_internal_path', 'dictionary_path']
            if next((x for x in project_choices if x in form_display), None):
                if 'project_id' not in form_display:
                    form_display.append('project_id')

            for options in form_display:
                output = study.loc[study['Label'] == options]
                result = pd.concat([result, output], axis = 0)

            if not result.empty:
                display_clean(result, path_display = True)


        # prints cleaned tables of all projects in study if specific data type was not chosen
        if not datatype:
            expand = st.checkbox(f'Display all projects included in the above study', key = study['Value'].iloc[1])
            if expand:
                for i in range(len(projects)-1):
                    datatype_project = study[projects[i]-1: projects[i+1]-1]
                    display_clean(datatype_project.reset_index(drop=True), study['Value'].iloc[projects[i]])

        # checks and prints projects with selected data types
        for i in range(len(projects)-1):
            if study['Value'].iloc[projects[i]-1] in datatype:
                    datatype_project = study[projects[i] - 1: projects[i+1] - 1]
                    display_clean(datatype_project.reset_index(drop=True), project_name = study['Value'].iloc[projects[i]], 
                    which_datatype = study['Value'].iloc[projects[i]-1])
        for type in datatype:
            if type not in list(study.loc[study['Label'] == 'data_type', 'Value']):
                st.info(f'{type} data type is not in this study') # info bar if selected data type not in study

# Import logos from Google Cloud
app_bucket_name = 'app_materials'
app_bucket = get_gcloud_bucket(app_bucket_name)
card_removebg = app_bucket.get_blob('card_removebg.png')
card_removebg = card_removebg.download_as_bytes()
diver_logo = app_bucket.get_blob('diver_logo.png')
diver_logo = diver_logo.download_as_bytes()
st.session_state['card_no_bg'] = card_removebg

# card_removebg = '/Users/kuznetsovn2/Desktop/diver_query/data/card_removebg.png'

# Configure Streamlit home page
st.set_page_config(
     page_title="Home",
     page_icon= card_removebg, 
     layout="centered",
)

# Import dataset XML + respective data dictionaries from RedCap
redcap = blob_as_xml(app_bucket, 'DIVER_CDISC_ODM.xml', ".//*")
diver = blob_as_csv(app_bucket, 'DIVER_DataDictionary.csv', sep = ',')

# redcap = pd.read_xml('data/DIVER_CDISC_ODM_2023-04-08_1610.xml', xpath=".//*")
# diver = pd.read_csv('data/DIVER_DataDictionary_2023-04-08.csv')

# Prepare data dictionary to map XML variables
diver_dict = diver[['Variable / Field Name', 'Choices, Calculations, OR Slider Labels']]
diver_dict.rename(columns = {'Variable / Field Name': 'variable_name',
                            'Choices, Calculations, OR Slider Labels': 'variable_options'}, inplace = True)
diver_dict.dropna(inplace = True)
diver_dict.set_index('variable_name', inplace = True)

# Prepare XML file for clean dataframe outputs
redcap_df = redcap[['StudyOID', 'SubjectKey', 'FormOID', 'ItemGroupOID', 'ItemOID', 'Value']]
redcap_df['StudyOID'] = redcap_df['StudyOID'].shift(4)
redcap_df['SubjectKey'] = redcap_df['SubjectKey'].shift(3)
redcap_df['FormOID'] = redcap_df['FormOID'].shift(2)
redcap_df['ItemGroupOID'] = redcap_df['ItemGroupOID'].shift(1)
redcap_df = redcap_df[redcap_df['Value'].notna()]
redcap_df.reset_index(drop=True, inplace=True)

# Map 'disease focus' values to disease names from dictionary
disease_vals = diver_dict.loc['disease_focus'].to_list()
disease_vals = disease_vals[0].split('|')

disease_dict = {} # extracts only disease values from dictionary
for item in disease_vals:
    item_list = item.split(',')
    disease_dict[item_list[0].strip()] = item_list[1].strip()

# More columns that must be mapped to dictionary values
map_these = ['overlap', 'study_timeframe', 'project_external_internal', 'dictionary', 'access', 'clinical_study_status',
            'imaging_type', 'imaging_derived_raw','availability_mmse', 'availability_moca', 'availability_neurocog', 
            'availability_med_status', 'data_type',]
replace_these = ['disease_focus', 'study_design', 'subjects', 'sex', 'race', 'ethnicity', 'imaging_type']

# Replaces variables with proper names in main dataframe
for col_name in map_these:
    datatype_vals = diver_dict.loc[col_name].to_list()
    datatype_vals = datatype_vals[0].split('|')

    datatype_dict = {}
    for item in datatype_vals:
        item_list = item.split(',')
        datatype_dict[item_list[0].strip()] = item_list[1].strip()

    datatype_ints = redcap_df.loc[redcap_df['ItemOID'] == col_name, 'Value']
    redcap_df.loc[redcap_df['ItemOID'] == col_name, 'Value'] = redcap_df.loc[redcap_df['ItemOID'] == col_name, 'Value'].map(datatype_dict)

for col in replace_these:
    replace_vals(col, redcap_df)

# Find Study/Reference file starting and stopping indices[currently does not output reference files]
studies = redcap_df['FormOID'].loc[lambda x: x=='Form.study_information'].index.values # study start point in main dataframe
ref_files = redcap_df['FormOID'].loc[lambda x: x=='Form.reference_file_information'].index.values # ref file start point
ref_file_stop = redcap_df['ItemGroupOID'].loc[lambda x: x=='reference_file_information.reference_file_information_complete'].index.values

total_indices = list(studies) + list(ref_files) + list(ref_file_stop) # will help identify separate study/ref files
total_indices.append(len(redcap_df))
total_indices = sorted(total_indices)

#### Create App 

# Background color
css = app_bucket.get_blob('style.css')
css = css.download_as_string()

# f = open('data/style.css', "r")
# css = f.readlines()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Main title
st.markdown("<h1 style='text-align: center; color: #0f557a; font-family: Helvetica; '>DIVER REDCap Query Tool</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; '>① Get Started with These Search Options</h3>", unsafe_allow_html=True)

# Create sidebar for more selection options
sidebar1, sidebar2, sidebar3 = st.sidebar.columns([0.25,1,0.25])
st.sidebar.markdown("<h3 style='text-align: center; '>② Edit Your Main Page Outputs</h3>", unsafe_allow_html=True)

# sidebar2.image('data/diver_logo.png', use_column_width=True)
sidebar2.image(diver_logo, use_column_width=True)
selectbox_options = list(disease_dict.values())
selectbox_options.insert(0, ' ') # defaults drop-down menu to "Choose an option"

# Create multiselect and automatically put it in state with its key parameter
form_display = st.sidebar.multiselect( # selects only the options DIVER Team would like to see most
'Paths to display:',
['access_path','project_external_path', 'project_internal_path', 'dictionary_path'], key = 'multiselect1')
datatype = st.sidebar.multiselect( # outputs projects based on their data types
'Data types to display:',
datatype_dict.values(), key = 'multiselect2')

# Button clears the state of the multiselect
st.sidebar.button("Clear Selections", on_click=clear_multi)

# Gives user option to choose studies based on their names or target disease
choice = st.radio(
    "Choose if you want to select by study name or disease",
    ('Study Name Search', 'Target Disease Selection'), label_visibility='collapsed', horizontal=True) # choose search option
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
study_names = list(redcap_df.loc[redcap_df['ItemOID'] == 'study_name', 'Value'])
study_names.insert(0, ' ')
if choice == 'Study Name Search':
    study_choice = st.selectbox('Study Name:', study_names)
    disease = st.selectbox('Target Disease:', selectbox_options, disabled = True)
    disease = None # clears output space
else:
    study_choice = st.selectbox('Study Name:', study_names, disabled = True)
    disease = st.selectbox('Target Disease:', selectbox_options)
    study_choice = None

start = 0
stop = 1

# Iterates through study start and stop indices to separate studies
while total_indices[start] < len(redcap_df):
    item_oids = []
    item_vals = []
    study = pd.DataFrame()

    for i in range(total_indices[start], total_indices[stop]): # holds single study
        item_oids.append(redcap_df['ItemOID'].iloc[i])
        item_vals.append(redcap_df['Value'].iloc[i])

    study['Label'] = item_oids
    study['Value'] = item_vals

    process_forms(study, disease, study_choice, datatype)

    start += 1
    stop += 1