from csv import reader
from math import exp
from random import shuffle
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
from math import log10

##Import data from file---------------------------------------------------------------------------------------------------------------------
def import_data(file_name):
    dataset = list()
    with open(file_name,'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    return dataset


##convert input string to float---------------------------------------------------------------------------------------------------------------------
def str_to_float(dataset,column):
    for row in dataset:
        row[column] = float(row[column].strip())

##convert string type to int type ---------------------------------------------------------------------------------------------------------------------
def str_to_int(dataset,column):

    for row in dataset:
        if row[column] == "Iris-setosa":
            del row[column]
            a=[0,0,1]
            row +=a
        elif row[column] == "Iris-versicolor":
            del row[column]
            a=[0,1,0]
            row+=a
        else:
            del row[column]
            a=[1,0,0]
            row+=a

def sigmoid(x):
    return (1/(1+exp(-x)))

def d_sigmoid(y):
    return (y*(1-y))

def activation(row,weights,bias,neuron):
    activation_list = list()
    a=0
    for i in range(neuron):
        activ = bias[i]
        for j in range(len(row)):            
            activ += weights[j+a] * row[j]
        a+=len(row)
        activation_list.append(sigmoid(activ))
    
    return activation_list

##find dwaight / update weight ---------------------------------------------------------------------------------------------------------------------
def dweight(err,inp,activ,weights,bias):
    dweight_list = list()
    dbias_list = list()
##    dweight
    for i in range(len(inp)):
        for j in range(len(activ)):
            dweight = 0
            dweight = err[j]*d_sigmoid(activ[j])*inp[i]
            dweight_list.append(dweight)
##  dbias
    for j in range(len(activ)):
        dbias = 0
        dbias = err[j]*d_sigmoid(activ[j])
        dbias_list.append(dbias)

##  updating dweight and dbias
    for k in range(len(dweight_list)):
        dweight_list[k] = weights[k]-l_rate*dweight_list[k]
        
    for k in range(len(dbias_list)):
        dbias_list[k] = bias[k]-l_rate*dbias_list[k]

    return (dweight_list,dbias_list)

##function multilayer ---------------------------------------------------------------------------------------------------------------------
def multilayer_perceptron(dataset):
    
    shuffle(dataset)
    
    data_train = dataset[:120]
    data_validation = dataset[120:]

    weights_ih, bias_ih, weight_ho, bias_ho = train(data_train)
    validation(data_validation,weights_ih, bias_ih, weight_ho, bias_ho)
    
    averrage()
     

def prediction(row,output):
    predict = 0
    for i in range(len(output)):
        if output[i] > 0.5:
           output[i]=1
        else:
            output[i]=0
    
    for i in range(len(output)):
        if output[i]==row[i]:
           predict = 1
        else:
            predict = 0
            
    return predict

def error(row,output):
    err_list = list()
    
    for i in range(len(output)):
        err =(1/2)*((row[i]-output[i])**2)
        err_list.append(err)
        
    return err_list

def error_hiden(error_o,weights_o):
    error_h = list()
    
    for i in range(len(error_o)):
        err_h = 0
        a = int(i*len(error_o))
        b = int(a+(len(error_o)))
        l = 0
        for j in range(a,b):
            err_h += weights_o[j]*error_o[l]
            l+=1
        error_h.append(err_h)

    return error_h

def to_matrix(weight):
    list_list = [[0 for i in range(3)] for j in range(perceptron)]
    for i in range(3):
        a=0
        for j in range(perceptron):
            list_list[i][j]= weight[j+a]
        a+= perceptron

    return list_list

def to_list(matrix):
    flat_list = [item for sublist in matrix for item in sublist]
    return flat_list

def tranpose(weight):
##    matrix = to_matrix(weight)
    list_temp = [ 1 for j in range(len(weight))]
    
    list_temp[0]=weight[0]
    list_temp[1]=weight[3]
    list_temp[2]=weight[6]
    list_temp[3]=weight[1]
    list_temp[4]=weight[4]
    list_temp[5]=weight[7]
    list_temp[6]=weight[2]
    list_temp[7]=weight[5]
    list_temp[8]=weight[8]
    
##    matrix = np.array(matrix).T.tolist()
##    weight = to_list(matrix)
    
    return list_temp

## train the data---------------------------------------------------------------------------------------------------------------------       
def train(data_train):
    
    weights_ih = [uniform(0.0,1.0) for i in range((len(dataset[0])-3)*perceptron)]
    bias_ih = [uniform(0.0,1.0) for i in range(perceptron)]
                
    weights_ho = [uniform(0.0,1.0) for i in range(perceptron*3)]
    bias_ho = [uniform(0.0,1.0) for i in range(3)]
    
    for i in range(epoch):
        sumAccuracy = 0
        sumError = 0
        
        for row in data_train:
            ##feedforward
            output_h = activation(row[:4],weights_ih,bias_ih,perceptron)
            output_o = activation(output_h,weights_ho,bias_ho,3)
            
            err = error(row[4:],output_o)
            sumError += sum(err)
            sumAccuracy += prediction(row[4:],output_o)

            ##backpropagation
            t_weights_ho = tranpose(weights_ho)
            error_h = error_hiden(err,t_weights_ho)
            
            weights_ho, bias_ho = dweight(err,output_h,output_o,weights_ho,bias_ho)

            weights_ih,bias_ih = dweight(error_h,row[:4],output_h,weights_ih,bias_ih)


        error_train.append(sumError/len(data_train))
        accuracy_train.append(sumAccuracy/len(data_train))

    return (weights_ih,bias_ih,weights_ho,bias_ho)

    
##validation the data after training---------------------------------------------------------------------------------------------------------------------
def validation(data_validation,weights_ih,bias_ih,weights_ho,bias_ho):
    for i in range(epoch):
        sumAccuracy = 0
        sumError = 0
        for row in data_validation:
            ##feedforward
            output_h = activation(row[:4],weights_ih,bias_ih,perceptron)
            output_o = activation(output_h,weights_ho,bias_ho,3)
            
            err = error(row[4:],output_o)
            sumError += sum(err)
            sumAccuracy += prediction(row[4:],output_o)

        error_validation.append(sumError/len(data_validation))
        accuracy_validation.append(sumAccuracy/len(data_validation))

def averrage():
    for i in error_train:
        i = log10(i)
    for j in accuracy_train:
        j = log10(j)

    for k in error_validation:
        k = log10(k)
    for l in accuracy_validation:
        l = log10(l)

    draw_grafik(error_train,error_validation,'Averrage Error','Error','Grafik Error Multilayer Perceptron')
    draw_grafik(accuracy_train,accuracy_validation,'Acuracy Error','Error','Grafik Acuracy Multilayer Perceptron')

##draw grafik ---------------------------------------------------------------------------------------------------------------------
def draw_grafik(data_train,data_validation,label,ylabel,title):
    
    plt.plot(data_train,label = label+' Train')
    plt.plot(data_validation, label = label+' Validation')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()   

##Main program ---------------------------------------------------------------------------------------------------------------------
dataset = import_data("iris_3_class.csv")
for i in range(len(dataset[0])-1):
    str_to_float(dataset,i)
str_to_int(dataset,len(dataset[0])-1)

l_rate = 0.1
#l_rate = 0.8
perceptron = 3
epoch = 300

error_train = list()
error_validation = list()
accuracy_train = list()
accuracy_validation =list()

multilayer_perceptron(dataset)





