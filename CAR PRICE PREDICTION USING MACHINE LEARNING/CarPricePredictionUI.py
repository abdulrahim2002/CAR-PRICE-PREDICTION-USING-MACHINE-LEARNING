# Author: Abdul Rahim(C) 2022

import pygame
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# ML essentials
#import required libraries
import pandas as pd

# to split data in training and testing
from sklearn.model_selection import train_test_split

# standard scaler to normalise the data
from sklearn.preprocessing import StandardScaler

# importing linear regression, decision tree regression and random forest regression 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor    

import numpy as np

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

# UI code starts here
pygame.init()
pygame.font.init()

clock = pygame.time.Clock()

parentDirectoryPath = os.getcwd()

# DEFINE SCENES
SCENE_UI = 0
SCENE_PREDICTION = 1
FULLSCREEN = 2
SCENE_STATE = SCENE_UI
WARNING = False
DIMENTION = -1

# OFFSET
HORIZONTAL_OFFSET = 0
VERTICAL_OFFSET = 0

# screen DEFAULT setup
DEFAULT_WIDTH = 1533
DEFAULT_HEIGHT = 810
DEFAULT = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
screen = pygame.display.set_mode(DEFAULT)

# font width and height
TITLE_FONT_WIDTH = 415
TITLE_FONT_HEIGHT = 30
SUBTITLE_FONT_WIDTH = 755
SUBTITLE_FONT_HEIGHT = 170
PARAGRAPH_FONT_WIDTH = 50
# PARAGRAPH_FONT_HEIGHT = [350,375 ..]
INPUT_FONT_WIDTH = 805
# INPUT_FONT_HEIGHT = [400,450 ..]

# defining the fonts
title = pygame.font.Font( f'{parentDirectoryPath}/fonts/OPTIBodoni-Antiqua.otf', 130)
subtitle = pygame.font.Font(f'{parentDirectoryPath}/fonts/Didot Regular.ttf', 20)
# paragraph font
paragraph = pygame.font.Font(f'{parentDirectoryPath}\\fonts\\Bodoni MT.TTF', 20)
# paragraph heading
paragraph_heading = pygame.font.Font(f'{parentDirectoryPath}\\fonts\\Bodoni MT.TTF', 50)
paragraph_heading.set_underline(True)
# input font
input_font = pygame.font.Font(f'{parentDirectoryPath}\\fonts\\Granjon.otf', 30)
# font for numbers entered
user_input_font = pygame.font.Font(f'{parentDirectoryPath}\\fonts\\jaguar.ttf', 20)
# font for prediction page
prediction_font = pygame.font.Font(f'{parentDirectoryPath}\\fonts\\Helvetica.ttf', 25)
# gets active when input box is clicked by user
color_active = pygame.Color('gray')
color_passive = pygame.Color('black')      

class inputRectangles:
    rectYear =          pygame.Rect(1255,370, 200, 35)
    colorYear =         color_passive
    
    rectPresent_Price = pygame.Rect(1255,415, 200, 35)
    colorPresent_Price= color_passive
    
    rectKms_Driven =    pygame.Rect(1255,460, 200, 35)
    colorKms_Driven =   color_passive
    
    rectFuel_Type =     pygame.Rect(1255,505, 200, 35)
    colorFuel_Type   =  color_passive

    rectSeller_Type =   pygame.Rect(1255,550, 200, 35)
    colorSeller_Type =  color_passive

    rectTransmission =  pygame.Rect(1255,595, 200, 35)
    colorTransmission = color_passive

    rectOwner =         pygame.Rect(1255,640, 200, 35)
    colorOwner =        color_passive

class inputStrings:
    Year = ''
    Present_Price = ''
    Kms_Driven = ''
    Fuel_Type = ''
    Seller_Type = ''
    Transmission = ''
    Owner = ''

class active:
    Year = False
    Present_Price = False
    Kms_Driven = False
    Fuel_Type = False
    Seller_Type = False
    Transmission = False
    Owner = False

class prediction:
    LinearRegression = 0
    PolynomialRegression = 0
    DecisionTree = 0
    RandomForest = 0

# title and icon
pygame.display.set_caption("Car Price Prediction")
icon = pygame.image.load(f'{parentDirectoryPath}/route66.png')
icon = pygame.transform.scale(icon, (32, 32))
pygame.display.set_icon(icon)
# loading buttons
predict_button = pygame.image.load(f'{parentDirectoryPath}/predictButton.png').convert_alpha()
predict_button = pygame.transform.scale(predict_button, (150, 57))
# predict button dimensions = drawx + 150, drawy + 55  

def fullscreen(current_screen):
    # change the screen
    if (current_screen.get_width(), current_screen.get_height()) == DEFAULT:
        screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(DEFAULT)

def write_contentUI():
    # draw predict button
    screen.blit(predict_button, (1230, 685))
    
    # draw line
    pygame.draw.line(screen, (0,0,0), (SUBTITLE_FONT_WIDTH, 300),(SUBTITLE_FONT_WIDTH, SUBTITLE_FONT_HEIGHT + 500), 1)
    
    # title font
    title_img = title.render('CLASSIFIED', True, (0, 0, 0))
    screen.blit(title_img, (TITLE_FONT_WIDTH, TITLE_FONT_HEIGHT))
    
    # SUBTITLE FONT
    subtitle_img = subtitle.render('KNOW THE PRICE OF YOUR CAR', True, (0, 0, 0))
    screen.blit(subtitle_img, (SUBTITLE_FONT_WIDTH, SUBTITLE_FONT_HEIGHT))

    # paragraph heading
    paragraph_heading_img = paragraph_heading.render('DESCRIPTION', True, (0, 0, 0))
    screen.blit(paragraph_heading_img, (PARAGRAPH_FONT_WIDTH, PARAGRAPH_FONT_HEIGHT[0] - 60))

    # paragraph font
    j = 0
    for line in content:
        paragraph_img = paragraph.render(line, True, (0, 0, 0))
        screen.blit(paragraph_img, (PARAGRAPH_FONT_WIDTH, PARAGRAPH_FONT_HEIGHT[j]))
        j += 1

    # input text
    input_text = paragraph_heading.render('Enter your car details', True, (0, 0, 0))
    screen.blit(input_text, (SUBTITLE_FONT_WIDTH + 50, PARAGRAPH_FONT_HEIGHT[0] - 60))

    k = 0
    for query in input:
        input_img = input_font.render( query , True, (0, 0, 0))
        screen.blit(input_img, (SUBTITLE_FONT_WIDTH + 50, INPUT_FONT_HEIGHT[k]))
        k += 1

    if WARNING:
        # warnign text
        warning_text = input_font.render('INVALID INPUT!', True, (255, 0, 0))
        screen.blit(warning_text, (SUBTITLE_FONT_WIDTH + 50, 700))

# paragraph content
content = [
    '',
    'This project aims to predict the price of a car using machine learning',
    'techniques. The project uses historical data on car sales and features',
    'to train a model that can predict  the price of a car given its charac-',
    'teristics. The model can be useful for determining the value of a used ',
    'car.',
    '',
    'The Machine Learning Algorithms used are: ',
    '1. Linear Regression',
    '2. Polynomial Regression',
    '3. Decision Tree Regression',
    '4. Random Forest Regression'
    ]
PARAGRAPH_FONT_HEIGHT = [x*25 + 350 for x in range(len(content))]

input = [
    'Year:',
    'New Price(in lakhs):',
    'Kms Driven(in km):',
    'Fuel Type(Petrol/Diesel/CNG):',
    'Seller Type(Dealer/Individual)):',
    'Transmission(Manual/Automatic):',
    'No. of previous owner:',
]
INPUT_FONT_HEIGHT = [x*45 + 370 for x in range(len(input))]

def write_contentPred():
    # title font
    title_img = title.render('PREDICTIONS', True, (0, 0, 0))
    screen.blit(title_img, (365, TITLE_FONT_HEIGHT))
    
    # SUBTITLE FONT
    subtitle_img = subtitle.render('THE PRICE OF YOUR CAR IS', True, (0, 0, 0))
    screen.blit(subtitle_img, (SUBTITLE_FONT_WIDTH - 130, SUBTITLE_FONT_HEIGHT))

    # ML ALGORITHM
    paragraph_heading_img = paragraph_heading.render('ML ALGORITHM', True, (0, 0, 0))
    screen.blit(paragraph_heading_img, (70, 290))

    # draw a line between ML ALGORITHM and PREDICTION
    pygame.draw.line(screen, (0,0,0), (PARAGRAPH_FONT_WIDTH + 475, 300),(PARAGRAPH_FONT_WIDTH + 475, SUBTITLE_FONT_HEIGHT + 500), 1)

    # PREDICTION
    paragraph_heading_img = paragraph_heading.render('PREDICTION', True, (0, 0, 0))
    screen.blit(paragraph_heading_img, (PARAGRAPH_FONT_WIDTH + 540, 290))

    # draw a line between PREDICTION and ACCURACY
    pygame.draw.line(screen, (0,0,0), (PARAGRAPH_FONT_WIDTH + 930, 300),(PARAGRAPH_FONT_WIDTH + 930, SUBTITLE_FONT_HEIGHT + 500), 1)

    # ACCURACY
    paragraph_heading_img = paragraph_heading.render('ACCURACY', True, (0, 0, 0))
    screen.blit(paragraph_heading_img, (PARAGRAPH_FONT_WIDTH + 1000, 290))

    i = 0
    for algorithm in algorithms:
        algorithm_img = prediction_font.render(algorithm, True, (0, 0, 0))
        screen.blit(algorithm_img, (PARAGRAPH_FONT_WIDTH + 20, ALGORITHM_FONT_HEIGHT[i]))
        i += 1

    width = 4
    predictions_list = [
    f'{prediction.LinearRegression:>{width}} Lakhs',
    f'{prediction.PolynomialRegression:>{width}} Lakhs',
    f'{prediction.DecisionTree:>{width}} Lakhs',
    f'{prediction.RandomForest:>{width}} Lakhs'
    ]

    j = 0
    for str in predictions_list:
        prediction_img = prediction_font.render(str, True, (0, 0, 0))
        screen.blit(prediction_img, (590, ALGORITHM_FONT_HEIGHT[j]))
        j += 1

    k = 0
    for accuracy in accuracies:
        accuracy_img = prediction_font.render(accuracy, True, (0, 0, 0))
        screen.blit(accuracy_img, (PARAGRAPH_FONT_WIDTH + 1000, ALGORITHM_FONT_HEIGHT[k]))
        k += 1

algorithms = [
    'LINEAR  REGRESSION',
    'POLYNOMIAL  REGRESSION',
    'DECISION TREE REGRESSION',
    'RANDOM FOREST REGRESSION'
    ] 

accuracies = [
    '82.9%',
    '93.8%',
    '94.0%',
    '95.9%'
    ]
ALGORITHM_FONT_HEIGHT = [x*70 + 400 for x in range(len(algorithms))]

def inputYear(events,screen):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inputRectangles.rectYear.collidepoint(event.pos):
                active.Year = True
            else:
                active.Year = False

        if active.Year:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    inputStrings.Year = inputStrings.Year[:-1]
                else:
                    inputStrings.Year += event.unicode

    if active.Year:
        inputRectangles.colorYear = color_active
    else:
        inputRectangles.colorYear = color_passive

    # draw rectangle and argument passed which should be on screen
    pygame.draw.rect(screen, inputRectangles.colorYear, inputRectangles.rectYear, 2)

    text_surface = user_input_font.render(inputStrings.Year, True, 'black')
	
	# render at position stated in arguments
    screen.blit(text_surface, (inputRectangles.rectYear.x+5, inputRectangles.rectYear.y+5))
	
	# set width of textfield so that text cannot get
	# outside of user's text input
    inputRectangles.rectYear.w = max(100, text_surface.get_width()+10)   

def inputPresent_Price(events,screen):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inputRectangles.rectPresent_Price.collidepoint(event.pos):
                active.Present_Price = True
            else:
                active.Present_Price = False

        if active.Present_Price:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    inputStrings.Present_Price = inputStrings.Present_Price[:-1]
                else:
                    inputStrings.Present_Price += event.unicode

    if active.Present_Price:
        inputRectangles.colorPresent_Price = color_active
    else:
        inputRectangles.colorPresent_Price = color_passive

    # draw rectangle and argument passed which should be on screen
    pygame.draw.rect(screen, inputRectangles.colorPresent_Price, inputRectangles.rectPresent_Price, 2)

    text_surface = user_input_font.render(inputStrings.Present_Price, True, 'black')
    
    # render at position stated in arguments
    screen.blit(text_surface, (inputRectangles.rectPresent_Price.x+5, inputRectangles.rectPresent_Price.y+5))
    
    # set width of textfield so that text cannot get
    # outside of user's text input
    inputRectangles.rectPresent_Price.w = max(100, text_surface.get_width()+10)

def inputKms_Driven(events,screen):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inputRectangles.rectKms_Driven.collidepoint(event.pos):
                active.Kms_Driven = True
            else:
                active.Kms_Driven = False

        if active.Kms_Driven:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    inputStrings.Kms_Driven = inputStrings.Kms_Driven[:-1]
                else:
                    inputStrings.Kms_Driven += event.unicode

    if active.Kms_Driven:
        inputRectangles.colorKms_Driven = color_active
    else:
        inputRectangles.colorKms_Driven = color_passive

    # draw rectangle and argument passed which should be on screen
    pygame.draw.rect(screen, inputRectangles.colorKms_Driven, inputRectangles.rectKms_Driven, 2)

    text_surface = user_input_font.render(inputStrings.Kms_Driven, True, 'black')
    
    # render at position stated in arguments
    screen.blit(text_surface, (inputRectangles.rectKms_Driven.x+5, inputRectangles.rectKms_Driven.y+5))
    
    # set width of textfield so that text cannot get
    # outside of user's text input
    inputRectangles.rectKms_Driven.w = max(100, text_surface.get_width()+10)

def inputFuel_Type(events,screen):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inputRectangles.rectFuel_Type.collidepoint(event.pos):
                active.Fuel_Type = True
            else:
                active.Fuel_Type = False

        if active.Fuel_Type:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    inputStrings.Fuel_Type = inputStrings.Fuel_Type[:-1]
                else:
                    inputStrings.Fuel_Type += event.unicode

    if active.Fuel_Type:
        inputRectangles.colorFuel_Type = color_active
    else:
        inputRectangles.colorFuel_Type = color_passive

    # draw rectangle and argument passed which should be on screen
    pygame.draw.rect(screen, inputRectangles.colorFuel_Type, inputRectangles.rectFuel_Type, 2)

    text_surface = user_input_font.render(inputStrings.Fuel_Type, True, 'black')
    
    # render at position stated in arguments
    screen.blit(text_surface, (inputRectangles.rectFuel_Type.x+5, inputRectangles.rectFuel_Type.y+5))
    
    # set width of textfield so that text cannot get
    # outside of user's text input
    inputRectangles.rectFuel_Type.w = max(100, text_surface.get_width()+10)

def inputSeller_Type(events,screen):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inputRectangles.rectSeller_Type.collidepoint(event.pos):
                active.Seller_Type = True
            else:
                active.Seller_Type = False

        if active.Seller_Type:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    inputStrings.Seller_Type = inputStrings.Seller_Type[:-1]
                else:
                    inputStrings.Seller_Type += event.unicode

    if active.Seller_Type:
        inputRectangles.colorSeller_Type = color_active
    else:
        inputRectangles.colorSeller_Type = color_passive

    # draw rectangle and argument passed which should be on screen
    pygame.draw.rect(screen, inputRectangles.colorSeller_Type, inputRectangles.rectSeller_Type, 2)

    text_surface = user_input_font.render(inputStrings.Seller_Type, True, 'black')
    
    # render at position stated in arguments
    screen.blit(text_surface, (inputRectangles.rectSeller_Type.x+5, inputRectangles.rectSeller_Type.y+5))
    
    # set width of textfield so that text cannot get
    # outside of user's text input
    inputRectangles.rectSeller_Type.w = max(100, text_surface.get_width()+10)

def inputTransmission(events,screen):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inputRectangles.rectTransmission.collidepoint(event.pos):
                active.Transmission = True
            else:
                active.Transmission = False

        if active.Transmission:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    inputStrings.Transmission = inputStrings.Transmission[:-1]
                else:
                    inputStrings.Transmission += event.unicode

    if active.Transmission:
        inputRectangles.colorTransmission = color_active
    else:
        inputRectangles.colorTransmission = color_passive

    # draw rectangle and argument passed which should be on screen
    pygame.draw.rect(screen, inputRectangles.colorTransmission, inputRectangles.rectTransmission, 2)

    text_surface = user_input_font.render(inputStrings.Transmission, True, 'black')
	
	# render at position stated in arguments
    screen.blit(text_surface, (inputRectangles.rectTransmission.x+5, inputRectangles.rectTransmission.y+5))
	
	# set width of textfield so that text cannot get
	# outside of user's text input
    inputRectangles.rectTransmission.w = max(100, text_surface.get_width()+10)   

def inputOwner(events,screen):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if inputRectangles.rectOwner.collidepoint(event.pos):
                active.Owner = True
            else:
                active.Owner = False

        if active.Owner:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    inputStrings.Owner = inputStrings.Owner[:-1]
                else:
                    inputStrings.Owner += event.unicode

    if active.Owner:
        inputRectangles.colorOwner = color_active
    else:
        inputRectangles.colorOwner = color_passive

    # draw rectangle and argument passed which should be on screen
    pygame.draw.rect(screen, inputRectangles.colorOwner, inputRectangles.rectOwner, 2)

    text_surface = user_input_font.render(inputStrings.Owner, True, 'black')
    
    # render at position stated in arguments
    screen.blit(text_surface, (inputRectangles.rectOwner.x+5, inputRectangles.rectOwner.y+5))
    
    # set width of textfield so that text cannot get
    # outside of user's text input
    inputRectangles.rectOwner.w = max(100, text_surface.get_width()+10)
    
def isValid():
    # empty
    if inputStrings.Year == '' or inputStrings.Present_Price == '' or inputStrings.Kms_Driven == '' or inputStrings.Fuel_Type == '' or inputStrings.Seller_Type == '' or inputStrings.Transmission == '' or inputStrings.Owner == '':
        return False
    
    # not valid datatype
    if inputStrings.Year.isdigit() == False or inputStrings.Kms_Driven.isdigit() == False or inputStrings.Fuel_Type.isalpha() == False or inputStrings.Seller_Type.isalpha() == False or inputStrings.Transmission.isalpha() == False or inputStrings.Owner.isdigit() == False:
        return False

    if (inputStrings.Fuel_Type not in ['Petrol', 'Diesel','CNG']) or (inputStrings.Seller_Type not in ['Dealer', 'Individual']) or (inputStrings.Transmission not in ['Manual', 'Automatic']):
        return False

    return True

def predict():
    input_query = []

    # year
    input_query.append(int(inputStrings.Year))

    # new price
    input_query.append(float(inputStrings.Present_Price))
    
    # kms driven
    input_query.append(int(inputStrings.Kms_Driven))

    # fuel type: is either Petrol or Diesel
    input_query.append(inputStrings.Fuel_Type)

    # Seller_Type: is either Dealer or Individual
    input_query.append(inputStrings.Seller_Type)
        
    # Transmission: is either Manual or Automatic
    input_query.append(inputStrings.Transmission)

    # Owner: is either 0, 1, 2 .. which represents the no. of previous owners
    input_query.append(int(inputStrings.Owner))

    X = input_query
    X = np.array(X)
    X = X.reshape(1, -1)
    X = pd.DataFrame(X, columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
    X.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)
    X = pd.get_dummies(X, columns=['Seller_Type', 'Transmission'], drop_first=True)
    X = X.reindex(columns=['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Owner', 'Seller_Type_Individual', 'Transmission_Manual'], fill_value=0)
    X = scaler.transform(X)

    # linear regression
    linearregressionprediction = model.predict(X)
    print(linearregressionprediction)
    prediction.LinearRegression = round(linearregressionprediction[0],2)

    # polynomial regression
    Xnew = poly_reg.fit_transform(X)
    polynomialregressionprediction = polynomialRegression.predict(Xnew)
    print(polynomialregressionprediction)
    prediction.PolynomialRegression = round(polynomialregressionprediction[0],2)

    # decision tree
    decisiontreeregression = DecisionTree.predict(X)
    prediction.DecisionTree = round(decisiontreeregression[0],2)
    print(decisiontreeregression)

    # random forest
    randomforestregression = randomForest.predict(X)
    prediction.RandomForest = round(randomforestregression[0],2)
    print(randomforestregression)

def drill():
    global car_data, X, y, X_train, X_test, y_train, y_test, scaler, model, predicted_price, poly_reg, X_poly, X_test_poly, polynomialRegression, y_pred_poly, regressor, y_predictSVR, DecisionTree, randomForest


    # importing the dataset
    car_data = pd.read_csv('car data.csv')

    # manual encoding
    car_data.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)

    #one hot encoding
    car_data = pd.get_dummies(car_data, columns=['Seller_Type', 'Transmission'], drop_first=True)

    X = car_data.drop(['Car_Name','Selling_Price'], axis=1)
    y = car_data['Selling_Price']

    # splitting the data into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

    # now we normalise our data using standard scaler
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # implementing Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # implementing Polynomial Regression Model
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X_train)
    X_test_poly = poly_reg.fit_transform(X_test)
    polynomialRegression = LinearRegression()
    polynomialRegression.fit(X_poly , y_train)

    # implementing decision tree Model
    DecisionTree = DecisionTreeRegressor(random_state = 0)
    DecisionTree.fit(X_train, y_train)

    # implementing random forest Model
    randomForest = RandomForestRegressor(n_estimators = 100)
    randomForest.fit(X_train, y_train)

# define all variables
drill()

# GAME LOOP
while True:
    screen.fill((240, 240, 240))
    events = pygame.event.get()
    for event in events:
        # check for quit
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        # check for full screen
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                fullscreen(screen)

        # if predict button is clicked
        if SCENE_STATE == SCENE_UI and event.type == pygame.MOUSEBUTTONDOWN and pygame.Rect((1230, 685), (150, 57)).collidepoint(event.pos):
            if isValid():
                predict()
                SCENE_STATE = SCENE_PREDICTION
            else:
                SCENE_STATE = SCENE_UI
                WARNING = True

    
    if SCENE_STATE == SCENE_UI:
        # content and buttons etc
        write_contentUI()
        # input
        inputYear(events,screen)
        inputPresent_Price(events,screen)
        inputKms_Driven(events,screen)
        inputFuel_Type(events,screen)
        inputSeller_Type(events,screen)
        inputTransmission(events,screen)
        inputOwner(events,screen)
    else:
        write_contentPred()
	
    pygame.display.update()
    clock.tick(10)