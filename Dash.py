# Import Library
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from dash.dependencies import Input, Output, State

rfc = pickle.load(open('model', 'rb'))
Scaler = pickle.load(open('scaler', 'rb'))
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Create layout
app.layout = html.Div(children=[ #children is to make child HTML tags
    html.H1('Will My Employee Leave Me?'), #H1 is for website header
    html.Div([
        html.Div([
            html.Div(id='output_age', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='age',
                min=18,
                max=60,
                step=1,
                marks={18: '18th', 60: '60 th'},
                value=30
            ),

            html.Div(id='output_distance', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='distance',
                min=0,
                max=30,
                step=1,
                marks={0: '0 km', 30: '30 km'},
                value=10
            ),
    
            html.Div(id='output_joblevel', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='joblevel',
                min=1,
                max=5,
                step=1,
                marks={1:'1', 2:'2', 3:'3', 4:'4', 5:'5'},
                value=2
            ),
    
            html.Div(id='output_numcompanies', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='NumCompanies',
                min=0,
                max=10,
                step=1,
                marks={0:'0', 10:'10'},
                value=3
            ),
    
            html.Div(id='output_totalworking', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='TotalWorking',
                min=0,
                max=40,
                step=1,
                marks={0:'0', 40: '40'},
                value=20
            ),
    
            html.Div(id='output_YearsInMyCompany', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='YearsInMyCompany',
                min=0,
                max=40,
                step=1,
                marks={0:'0', 40: '40'},
                value=20
            ),
   
            html.Div(id='output_TrainingTimes', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='TraningTimes',
                min=0,
                max=6,
                step=1,
                marks={0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6'},
                value=3
            ),
            html.Div(id='output_LastPromotion', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='LastPromotion',
                min=0,
                max=15,
                step=1,
                marks={0:'0', 15:'15'},
                value=3
            ),
 
            html.Div(id='output_CurrManager', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='CurrManager',
                min=0,
                max=20,
                step=1,
                marks={0:'0', 20:'20'},
                value=1
            ),
    
            html.Div(id='output_PercentSalaryHike', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='PercentSalaryHike',
                min=10,
                max=25,
                step=1,
                marks={10:'10',25:'25'},
                value=15
            ),
            html.Div(id='output_StockOption', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='StockOption',
                min=0,
                max=3,
                step=1,
                marks={0:'0', 1:'1', 2:'2', 3:'3'},
                value=2
            )
        ], className="six columns"),
        html.Div([
            html.Div(id='output_gender', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='gender',
                options=[
                    {'label': 'Female', 'value': 'Female'},
                    {'label': 'Male', 'value': 'Male'}
                ],
            ),  
            html.Div(id='output_MaritalStatus', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='MaritalStatus',
                options=[
                    {'label': 'Divorced', 'value': 'Divorced'},
                    {'label': 'Married', 'value': 'Married'},
                    {'label': 'Single', 'value': 'Single'}
                ], 
            ),  
            html.Div(id='output_education', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='education',
                options=[
                    {'label': 'Below College', 'value': '1'},
                    {'label': 'College', 'value': '2'},
                    {'label': 'Bachelor', 'value': '3'},
                    {'label': 'Master', 'value': '4'},
                    {'label': 'Doctor', 'value': '5'}
                ]
            ),
            html.Div(id='output_EducationField', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='EducationField',
                options=[
                    {'label': 'Human Resources', 'value': 'Human Resources'},
                    {'label': 'Life Sciences', 'value': 'Life Sciences'},
                    {'label': 'Marketing', 'value': 'Marketing'},
                    {'label': 'Medical', 'value': 'Medical'},
                    {'label': 'Technical Degree', 'value': 'Technical Degree'},
                    {'label': 'Other', 'value': 'Other'}
                ]
            ),
            html.Div(id='output_Department', style={'font-weight': 'bold'}),
            dcc.Dropdown(
            id='Department',
                options=[
                    {'label': 'Human Resources', 'value': 'Human Resources'},
                    {'label': 'Research & Development', 'value': 'Research & Development'},
                    {'label': 'Sales', 'value': 'Sales'}
                ]
            ),
            html.Div(id='output_JobRole', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='JobRole',
                options=[
                    {'label': 'Healthcare Representative', 'value': 'Healthcare Representative'},
                    {'label': 'Human Resources', 'value': 'Human Resources'},
                    {'label': 'Laboratory Technician', 'value': 'Laboratory Technician'},
                    {'label': 'Manager', 'value': 'Manager'},
                    {'label': 'Manufacturing Director', 'value': 'Manufacturing Director'},
                    {'label': 'Research Director', 'value': 'Research Director'},
                    {'label': 'Research Scientist', 'value': 'Research Scientist'},
                    {'label': 'Sales Executive', 'value': 'Sales Executive'},
                    {'label': 'Sales Representative', 'value': 'Sales Representative'},
                ]
            ),
            html.Div(id='output_FreqBusinessTravel', style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='FreqBusinessTravel',
                options=[
                    {'label': 'Never', 'value': 'Non_Travel'},
                    {'label': 'Rarely', 'value': 'Travel_Rarely'},
                    {'label': 'Frequently', 'value': 'Travel_Frequently'}
                ]
            ),
            html.Div(id='out-all-types', style={'font-weight': 'bold'}),
            dcc.Input(
                id="input_{}".format("number"),
                type="number",
                placeholder="Monthly Salary {}".format("number")
            ),
            #html.Button('Predict Me!', id='run_predict',n_clicks=0),
            html.H3(id='pred_result', style={'font-weight': 'bold'})
        ], className = "six columns"),
    ], className = "row")
   
])

@app.callback(
    Output('output_age', 'children'),
    [Input('age', 'value')])
def update_output(value):
    return 'Age: {} years'.format(value)
    
@app.callback(
    Output('output_distance', 'children'),
    [Input('distance', 'value')])
def update_output(value):
    return 'Distance From Home: {} km'.format(value)
    
@app.callback(
    Output('output_joblevel', 'children'),
    [Input('joblevel', 'value')])
def update_output(value):
    return 'Job Level: {}'.format(value)
    
@app.callback(
    Output('output_numcompanies', 'children'),
    [Input('NumCompanies', 'value')])
def update_output(value):
    return 'Total Number of Companies: {}'.format(value)
    
@app.callback(
    Output('output_totalworking', 'children'),
    [Input('TotalWorking', 'value')])
def update_output(value):
    return 'Total Years of Experience: {} years'.format(value)
    
@app.callback(
    Output('output_YearsInMyCompany', 'children'),
    [Input('YearsInMyCompany', 'value')])
def update_output(value):
    return 'Total Years Employee at My Company: {} years'.format(value)
    
@app.callback(
    Output('output_TrainingTimes', 'children'),
    [Input('TraningTimes', 'value')])
def update_output(value):
    return 'Total Training Time Employee at My Company: {}x'.format(value)

@app.callback(
    Output('output_LastPromotion', 'children'),
    [Input('LastPromotion', 'value')])
def update_output(value):
    return 'Total Years Since Last Promotion: {} years'.format(value)
    
@app.callback(
    Output('output_CurrManager', 'children'),
    [Input('CurrManager', 'value')])
def update_output(value):
    return 'Total Years with Current Manager: {} years'.format(value)

@app.callback(
    Output('output_PercentSalaryHike', 'children'),
    [Input('PercentSalaryHike', 'value')])
def update_output(value):
    return 'Percent Salary Hike Last Year: {}'.format(value)

@app.callback(
    Output('output_StockOption', 'children'),
    [Input('StockOption', 'value')])
def update_output(value):
    return 'Stock Option Level: {}'.format(value)
    
@app.callback(
    Output('output_gender', 'children'),
    [Input('gender', 'value')])
def update_output(value):
    return 'Gender: {}'.format(value)

@app.callback(
    Output('output_MaritalStatus', 'children'),
    [Input('MaritalStatus', 'value')])
def update_output(value):
    return 'Marital Status: {}'.format(value)
    
@app.callback(
    Output('output_education', 'children'),
    [Input('education', 'value')])
def update_output(value):
    return 'Education: {}'.format(value)  

@app.callback(
    Output('output_EducationField', 'children'),
    [Input('EducationField', 'value')])
def update_output(value):
    return 'Education Field: {}'.format(value)  

@app.callback(
    Output('output_Department', 'children'),
    [Input('Department', 'value')])
def update_output(value):
    return 'Department: {}'.format(value)  

@app.callback(
    Output('output_JobRole', 'children'),
    [Input('JobRole', 'value')])
def update_output(value):
    return 'Job Role: {}'.format(value)  
    
@app.callback(
    Output('output_FreqBusinessTravel', 'children'),
    [Input('FreqBusinessTravel', 'value')])
def update_output(value):
    return 'Frequency Business Travel Last Year: {}'.format(value)      

@app.callback(
    Output('out-all-types', 'children'),
    [Input("input_{}".format("number"), "value")])
def update_output(value):
    return 'Monthly Income: {} $'.format(value)

@app.callback(
    Output('pred_result', 'children'),
    [Input('age', 'value'),
    Input('distance', 'value'),
    Input('joblevel', 'value'),
    Input('NumCompanies', 'value'),
    Input('TotalWorking', 'value'),
    Input('YearsInMyCompany', 'value'),
    Input('TraningTimes', 'value'),
    Input('LastPromotion', 'value'),
    Input('CurrManager', 'value'),
    Input('PercentSalaryHike', 'value'),
    Input('StockOption', 'value'),
    Input('gender', 'value'),
    Input('MaritalStatus', 'value'),
    Input('education', 'value'),
    Input('EducationField', 'value'),
    Input('Department', 'value'),
    Input('JobRole', 'value'),
    Input('FreqBusinessTravel', 'value'),
    Input("input_{}".format("number"), 'value')])
  

def update_prediction(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19):
    #PREPROCESSING DATA INPUT
    #Age
    if (x1<=30):
        x1=1
    elif (x1>30) & (x1<=40):
        x1=2
    elif (x1>40) & (x1<=50):
        x1=3
    elif (x1>50) & (x1<=60):
        x1=4 
    #Total Number of Companies
    x4= float(x4)
    #Total Year Experience
    x5= float(x5)
    if (x12== 'Female'):
        x12=0
    elif (x12== 'Male'):
        x12=1
        
    if (x13== 'Divorced'):
        x13=0
    elif (x13== 'Married'):
        x13=1
    elif(x13=='Single'):
        x13=2
        
    if (x14== 'Below College'):
        x14=1
    elif (x14=='College'):
        x14=2
    elif (x14=='Bachelor'):
        x14=3
    elif (x14=='Master'):
        x14=4
    elif (x14=='Doctor'):
        x14=5
    
    if (x15== 'Human Resources'):
        x15=0
    elif (x15=='Life Sciences'):
        x15=1
    elif (x15=='Marketing'):
        x15=2
    elif (x15=='Medical'):
        x15=3
    elif (x15=='Other'):
        x15=4   
    elif (x15=='Technical Degree'):
        x15=5
   
    if (x16== 'Human Resources'):
        x16= 0
    elif (x16== 'Research & Development'):
        x16= 1
    elif (x16== 'Sales'):
        x16=2

    if (x17== 'Healthcare Representative'):
        x17=0
    elif (x17== 'Human Resources'):
        x17=1
    elif (x17== 'Laboratory Technician'):
        x17=2    
    elif (x17== 'Manager'):
        x17=3
    elif (x17== 'Manufacturing Director'):
        x17=4
    elif (x17== 'Research Director'):
        x17=5
    elif (x17== 'Research Scientist'):
        x17=6
    elif (x17== 'Sales Executive'):
        x17=7
    elif (x17== 'Sales Representative'):
        x17=8
    
    if (x18== 'Non_Travel'):
        x18=0
    elif (x18== 'Travel_Frequently'):
        x18=1    
    elif (x18== 'Travel_Rarely'):
        x18=2
    # Create a NumPy array in the form of the original features
    input_X = np.array([x1, #Age
                        x18, #BusinessTravel
                        x16, #Department
                        x2, #Distance From Home
                        x14, #Education
                        x15, #Education Field
                        x12, #Gender
                        x3, #Joblevel
                        x17, #JobRole
                        x13, #MaritalStatus
                        x19, #MonthlyIncome
                        x4, #Numcompanies
                        x10, #Percent Salary
                        x11, #StockOption
                        x5, #TOtal Working Years
                        x7, #Training Time Last Year
                        x6, #YearsAtCompany
                        x8, #Years since last promotion
                        x9]).reshape(1,-1)
    input_X = (input_X.astype(float))
    input_X = Scaler.transform(input_X)
    #print(np.mean(input_X),np.std(input_X))
    predictions= rfc.predict_proba(input_X)[0,1]
    predictions = predictions*100
    return 'Attrition Rate Prediction: {}%'.format(predictions)
    

if __name__ == '__main__':
    app.run_server(debug=True)